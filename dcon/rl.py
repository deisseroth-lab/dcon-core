import numpy as np
import torch as T
import torch.nn.functional as F
from multiprocessing.pool import ThreadPool
import logging
from dcon.util import Timer, kaiser2d

EPS = 1e-3

# Utility to get efficient FFT sizes
def fft_len(x):
    full = x - 1
    order = np.log2(full)
    if order <= 10:
        return int(2**np.ceil(order))
    else:
        scale = full // 512 + 1
        return 512 * scale

# RL algorithm implemented on multiple GPUs in pytorch
def ifftshift_torch(x):
    out = T.empty_like(x)
    qx, qy = [(s + 1) // 2 for s in x.shape[1:]]
    px, py = [s - p for s, p in zip(x.shape[1:], [qx, qy])]
    out[:,:qx,:qy] = x[:,-qx:,-qy:]
    out[:,:qx,-py:] = x[:,-qx:,:py]
    out[:,-px:,:qy] = x[:,:px,-qy:]
    out[:,-px:,-py:] = x[:,:px,:py]
    return out

def rfftshift_torch(x):
    out = T.empty_like(x)
    px, py = [(s + 1) // 2 for s in x.shape[1:-1]]
    qx, qy = [s - p for s, p in zip(x.shape[1:-1], [px, py])]
    out[:,:qx,:,:] = x[:,-qx:,:,:]
    out[:,-px:,:,:] = x[:,:px,:,:]
    return out

def irfftshift_torch(x):
    out = T.empty_like(x)
    qx, qy = [(s + 1) // 2 for s in x.shape[1:-1]]
    px, py = [s - p for s, p in zip(x.shape[1:-1], [qx, qy])]
    out[:,:qx,:,:] = x[:,-qx:,:,:]
    out[:,-px:,:,:] = x[:,:px,:,:]
    return out

def rfft_torch(x):
    return T.rfft(x, 2, normalized=True)

def irfft_torch(x):
    s = (x.shape[1], 2 * (x.shape[2] - 1))
    return T.irfft(x, 2, normalized=True, signal_sizes=s)

def conj(x):
    out = T.empty_like(x)
    out[...,0] = x[...,0]
    out[...,1] = -x[...,1]
    return out

def cmult(x, y):
    rx = x[...,0]
    ix = x[...,1]
    ry = y[...,0]
    iy = y[...,1]
    return T.stack([rx * ry - ix * iy, rx * iy + ix * ry], dim=-1)


class RLTorch:
    def __init__(self, psfs, supersamples, channels, mags, padding=0, batch_size=1):
        self.psfs = psfs # Ch x C
        Z, X, Y = psfs[0].shape
        S = supersamples
        Ch = channels
        C = len(mags) // Ch
        self.dims = (S, Ch, C, Z, X, Y)
        self.r = np.array(mags).reshape(Ch, C) / min(mags)
        self.padding = padding

        self.n_gpus = T.cuda.device_count()
        self.slabs = np.array_split(np.arange(Z), int(np.ceil(Z/batch_size)))
        logging.info("{:d} slabs".format(len(self.slabs)))

        Xf = X + padding * 2
        Yf = Y + padding * 2

        XA = np.round(Xf * (self.r - 1) / 2).astype(np.int)
        YA = np.round(Yf * (self.r - 1) / 2).astype(np.int)
        XS = np.round(Xf * (self.r - 1/self.r) / 2).astype(np.int)
        YS = np.round(Yf * (self.r - 1/self.r) / 2).astype(np.int)
        self.margins = (XA, YA, XS, YS)

        self.W = T.tensor(1.)
        
        T.backends.cuda.cufft_plan_cache.max_size = 0
        self.pool = ThreadPool(processes=self.n_gpus)

        self.otfs = self._compute_otfs()

    def _single_otfs(self, s, otf):
        S, Ch, C, Z, X, Y = self.dims
        Xf, Yf = X + 2 * self.padding, Y + 2 * self.padding
        oidx = np.arange(Ch * C).reshape(Ch, C)
        dev_id = s % self.n_gpus
        slab = self.slabs[s]

        for ch in range(Ch):
            for c in range(C):
                psf = T.from_numpy(self.psfs[oidx[ch,c]][slab]).to(dev_id, non_blocking=True)
                if self.padding > 0:
                    psf = F.pad(psf, ((self.padding, self.padding, self.padding, self.padding)))
                otf[ch,c] = rfft_torch(ifftshift_torch(psf))
        T.cuda.empty_cache()
        return otf

    def _compute_otfs(self):
        S, Ch, C, Z, X, Y = self.dims
        Xf, Yf = X + 2 * self.padding, Y + 2 * self.padding
        # Quirk: you have to allocate the OTFs in the main thread to avoid a race condition in context creation
        # See https://github.com/pytorch/pytorch/issues/16559
        otfs = [T.empty((Ch, C, len(self.slabs[s]), Xf, Yf//2 + 1, 2), dtype=T.float32, device=s%self.n_gpus) for s in range(len(self.slabs))]
        with Timer("OTFs"):
            otfs = self.pool.starmap(self._single_otfs, zip(range(len(self.slabs)), otfs), chunksize=1)
            T.cuda.synchronize()
        return otfs

    def _single_forward(self, volume, otfs):
        S, Ch, C, Z, X, Y = self.dims
        Xf, Yf = X + 2 * self.padding, Y + 2 * self.padding
        XA, YA, SX, YS = self.margins
        Xo, Yo = volume.shape[-2:]
        Xp, Yp = (Xf-Xo)//2, (Yf-Yo)//2
        dev_id = volume.device.index

        img_est = T.zeros((Ch, X//S, Y//S), dtype=T.float32, device=dev_id)
        W = self.W.to(dev_id)

        for ch in range(Ch):
            for c in range(C):
                xa = XA[ch,c]
                ya = YA[ch,c]
                if self.r[ch,c] != 1:
                    slab_fft = T.zeros((volume.shape[0], Xf+xa*2, Yf//2+ya + 1, 2), dtype=T.float32, device=dev_id)
                    slab_fft[:,xa:-xa,:-ya] = rfftshift_torch(W * rfft_torch(F.pad(volume, (Yp, Yp, Xp, Xp))))
                    slab_zoomed = irfft_torch(irfftshift_torch(slab_fft))[:,xa:-xa,ya:-ya]
                else:
                    slab_zoomed = F.pad(volume, (Yp, Yp, Xp, Xp))
                slab_F = cmult(otfs[ch,c], rfft_torch(slab_zoomed))
                # Crop to downsample to sensor size
                slab_Fs = rfftshift_torch(slab_F)
                if S > 1:
                    slab_Fs = slab_Fs[:,Xf*(S-1)//(2*S):-Xf*(S-1)//(2*S),:-Yf*(S-1)//(2*S),:]
                slab_sum = irfft_torch(irfftshift_torch(slab_Fs))
                if self.padding:
                    slab_sum = slab_sum[:,self.padding//S:-self.padding//S,self.padding//S:-self.padding//S]
                img_est[ch] += slab_sum.sum(0)

        return img_est.clamp(min=0)

    def _single_backward(self, volume, otfs, ratio):
        S, Ch, C, Z, X, Y = self.dims
        p = self.padding
        Xf, Yf = X + 2 * self.padding, Y + 2 * self.padding
        XA, YA, XS, YS = self.margins
        Xv, Yv = volume.shape[-2:]
        Xe, Ye = (Xf-Xv)//2, (Yf-Yv)//2
        Xr, Yr = X//S + p, Y//S + p
        Xb, Yb = (Xf-Xr)//2, (Yf-Yr)//2

        dev_id = volume.device.index
        
        FR = rfft_torch(F.pad(ratio.to(dev_id), (p//2, p//2, p//2, p//2)))
        del ratio
        FR = irfftshift_torch(F.pad(rfftshift_torch(FR), (0, 0, 0, Xb, Yb, Yb)))
        
        update = T.zeros_like(volume)
        W = self.W.to(dev_id)

        for ch in range(Ch):
            for c in range(C):
                xa = XA[ch,c]
                ya = YA[ch,c]
                xs = XS[ch,c]
                ys = YS[ch,c]
                if self.r[ch,c] != 1:
                    slab_fft = T.zeros((volume.shape[0], Xf+xa*2, Yf//2+ya + 1, 2), dtype=T.float32, device=dev_id)
                    slab_zoomed = T.zeros((volume.shape[0], Xf+xa*2, Yf+ya*2), dtype=T.float32, device=dev_id)
                    slab_fft[:,xa:-xa,:-ya] = rfftshift_torch(W * cmult(FR[ch], conj(otfs[ch,c])))
                    slab_zoomed[:,xs:-xs,ys:-ys] = irfft_torch(irfftshift_torch(slab_fft[:,xs:-xs,:-ys])).clamp(min=0)
                    slab_update = slab_zoomed[:,xa:-xa,ya:-ya]
                else:
                    slab_update = irfft_torch(cmult(FR[ch], conj(otfs[ch,c]))).clamp(min=0)

                update += slab_update[:,Xe:-Xe,Ye:-Ye]
          
        volume.mul_(update)
        
        return volume

    def forward_project(self, volume):
        """Project a volume into a predicted light field.

        Args:
            volume: known volume to project to a light field (Z, roi_size, roi_size)

        Note:
            This is a numpy-speaking function.
        """
        S, Ch, C, Z, X, Y = self.dims
        volume = T.as_tensor(volume)
        volume = self.distribute(volume)
        proj = self.forward(volume)
        return proj.to('cpu').numpy()

    def forward(self, volume):
        """Leaves the light field on the head GPU (0)"""
        S, Ch, C, Z, X, Y = self.dims
        img_est = self.pool.starmap(self._single_forward, zip(volume, self.otfs), chunksize=1)
        T.cuda.synchronize()
        img_est = T.cuda.comm.reduce_add(img_est, destination=0)
        # Rescaling factor
        #img_est *= Ch * X
        return img_est

    def backward(self, ratio, volume):
        S, Ch, C, Z, X, Y = self.dims
        volume = self.pool.starmap(lambda v, o: self._single_backward(v, o, ratio), zip(volume, self.otfs), chunksize=1)
        T.cuda.synchronize()
        return volume

    def kaiser_window(self, order):
        S, Ch, C, Z, X, Y = self.dims
        Xf, Yf = X + 2 * self.padding, Y + 2 * self.padding
        W = kaiser2d(Xf, order)[:,X//2-1:].astype(np.float32)
        W = T.from_numpy(W)[None,:,:,None]
        W = irfftshift_torch(W)
        return W

    def iterate(self, img_exp, volume):
        with Timer("Forward"):
            img_est = self.forward(volume)

        with Timer("Ratio"):
            #eps = img_est.max() * EPS
            #eps = EPS
            eps = 1e-9
            ratio = img_exp / (img_est + eps)

        with Timer("Backward"):
            volume = self.backward(ratio, volume)

        return volume
    
    def distribute(self, volume):
        return [volume[slab].to(d%self.n_gpus) for (d, slab) in enumerate(self.slabs)]
    
    def collect(self, volume):
        return T.cat([v.to('cpu') for v in volume], dim=0)

    def deconvolve(self, image, roi_size, n_iters, kaiser_ord=0):
        """Perform a deconvolution.

        Args:
            image: light field collected (Ch x X x Y)
            roi_size: width/height of the final volume
            n_iters: RL iterations to perform

        Note:
            This is a numpy-speaking function.
        """
        with Timer("Deconvolution"):
            S, Ch, C, Z, X, Y = self.dims
            image = T.as_tensor(image).to(0)
            volume = T.ones((Z, roi_size, roi_size), dtype=T.float32)
            volume = self.distribute(volume)
            if kaiser_ord:
                self.W = self.kaiser_window(kaiser_ord)
            for itr in range(n_iters):
                with Timer("Iter {:d}".format(itr)):
                    volume = self.iterate(image, volume)
            return self.collect(volume).numpy()
