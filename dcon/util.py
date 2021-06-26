import numpy as np
import logging
from os.path import split, splitext
import time


def pos(x):
    return np.maximum(x, 0)

def kaiser2d(N, beta):
    from scipy.special import i0
    dline = np.arange(-N//2,N//2)
    Xp, Yp = np.meshgrid(dline, dline)
    D = np.hypot(Xp, Yp)
    alpha = N / 2.0
    w = i0(beta * np.sqrt(1 - ((D) / alpha)**2)) / i0(beta)
    w = np.nan_to_num(w)
    return w

def get_psf_info(psf_fnames):
    psf_locs = np.stack([splitext(split(fname)[1])[0][-5:].split('_') for fname in psf_fnames]).astype(np.int)
    Sx, Sy, C = np.max(psf_locs, axis=0) + 1
    return Sx, Sy, C

def max_inds(A):
    return np.unravel_index(A.argmax(), A.shape)

def radial_profile(dist, bins=10):
    from scipy.ndimage import mean as ndmean
    xs, ys = dist.shape
    nbins = bins
    Xs, Ys = np.ogrid[0:xs,0:ys]
    r = np.hypot(Xs - xs/2, Ys - ys/2)
    rbin = (nbins * r/r.max()).astype(np.int)
    bins = np.arange(1, rbin.max() + 1)
    rmean = ndmean(dist, labels=rbin, index=bins)
    _, edges = np.histogram(r, bins=nbins)
    return rmean, edges

def block_dilate(arr, selem, block_size, pool=None):
    from joblib import Parallel, delayed
    from itertools import product
    from scipy.ndimage import binary_dilation

    assert arr.ndim == selem.ndim

    Ndims = arr.ndim
    if isinstance(block_size, int):
        block_size = Ndims*(block_size,)
    if pool is None:
        pool = Parallel(n_jobs=-1, verbose=4)
    ol = [d//2 for d in selem.shape] # overlap

    indices = []
    for dim in range(Ndims):
        lower = np.arange(0, arr.shape[dim], block_size[dim])
        upper = lower + block_size[dim]
        lower -= ol[dim]
        upper += ol[dim]
        indices.append((np.maximum(0, lower), np.minimum(arr.shape[dim], upper)))

    slices =[[np.s_[l:u] for (l, u) in zip(ls, us)] for (ls, us) in indices]

    parts = pool([delayed(binary_dilation)(arr[s], selem) for s in product(*slices)])

    out = np.zeros_like(arr)
    for s, p in zip(product(*slices), parts):
        out[s] = np.maximum(out[s], p)

    return out

def crop_around(arr, center, margin, include_center=False):
    if not isinstance(margin, list):
        margin = [margin]*len(center)
    return arr[tuple(np.s_[c-m:c+m+int(include_center)] for (c, m) in zip(center, margin))]

def crop_center(arr, widths):
    mid = [s // 2 for s in arr.shape]
    hw = [w // 2 for w in widths]
    return crop_around(arr, mid, hw)

def bin_pixels(image, factor):
    from itertools import product
    result = sum([image[...,i::factor,j::factor] for (i, j) in product(range(factor), range(factor))]) / factor**2
    return result.astype(image.dtype)

def make_blobs(shape, n_blobs, mean_intensity, dxy, dz, sigmas):
    from scipy.ndimage.filters import gaussian_filter
    C = 0
    for n, sigma in zip(n_blobs, sigmas):
        A = np.zeros(shape, dtype=np.float32)
        pos_idx = np.random.choice(np.prod(shape), size=n, replace=False)
        S = np.unravel_index(pos_idx, shape)
        A[S] = sigma**2 * np.random.poisson(mean_intensity, n)
        A = gaussian_filter(A, sigma/np.array([dz, dxy, dxy]))
        C += A
    return C

def reshape_for_channels(raw, channels):
    """Transforms a 3D stack with interleaved channels into a 4D hyperstack with a leading channel dimension, 
    and also flips the 2nd channel. Useful for handing the multicamera PSF stacks saved by MicroManager."""
    assert channels in (1, 2), "Only 1 or 2 channels implemented"
    
    if channels == 2:
        assert raw.ndim in (3, 4), "If 2 channels, input must be 3 or 4D"
        
        if raw.ndim == 3:
            Z, X, Y = raw.shape
            Z = Z // channels
            raw = raw.reshape(Z, channels, X, Y)
        
        assert channels in raw.shape[:2], "Failed to form channel dimension properly"
        
        if raw.shape[1] == channels:
            raw = raw.swapaxes(0, 1)
        
        # now in Ch x Z x X x Y order
        # flip the second channel
        raw[1] = raw[1,:,:,::-1]
        return raw
    elif channels == 1:
        # Prepend a singleton dimension to the front
        return raw[None,...]


class Timer:
    def __init__(self, step_name):
        self.step_name = step_name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.t1 = time.time()
        self.dt = self.t1 - self.t0
        logging.info("[{:0.1f}s] {:s}".format(self.dt, self.step_name))
