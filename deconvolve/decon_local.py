#!/usr/bin/env python
import argparse as ap
import numpy as np
from os.path import basename, join, dirname
from s3tools import get_keys, get_s3_img, put_s3_img
from tqdm import tqdm
from joblib import Parallel, delayed
from rl import RLTorch


CAM_OFFSET = 100


def main(in_prefixes, psf_prefix, obj_xy, mags, n_iters, darkframe_prefix=None, out_dir='decon', padding=0,
         kaiser_order=0.0, z_batch=8, restart=False):
    """Perform a set of deconvolutions"""
    # Get image keys and extract some shape information from a test frame
    in_keys = []
    for prefix in in_prefixes:
        in_keys.extend([k for k in get_keys(prefix) if k[-4:] == ".tif"])
    test_img = get_s3_img(in_keys[0])
    img_shape = test_img.shape
    del test_img
    Ch = img_shape[0]
    print(f"{len(in_keys):d} images to deconvolve")

    # Get PSF paths and enough information to get all the array shapes right
    psf_keys = get_keys(psf_prefix)
    C = len(psf_keys) // Ch
    print(f"Found {Ch:d} channels and {C:d} classes")

    # Get darkframe if available
    if darkframe_prefix is not None:
        darkframe_key = get_keys(darkframe_prefix)[0]
        darkframe = get_s3_img(darkframe_key).astype(np.float32)
    else:
        darkframe = CAM_OFFSET

    # Fetch PSF data
    with Parallel(n_jobs=len(psf_keys), backend='threading') as pool:
        psfs = pool(delayed(get_s3_img)(key) for key in psf_keys)

    S = psfs[0].shape[-1] // img_shape[-1]
    rl = RLTorch(psfs, S, Ch, mags, padding=padding, batch_size=z_batch)

    for image_fname in tqdm(in_keys):
        # Determine output key
        if out_dir is None:
            final_out_dir = dirname(image_fname)
        else:
            final_out_dir = join(dirname(image_fname), out_dir)

        out_key = join(final_out_dir, "decon_" + basename(image_fname))

        # If in restart mode and the output exists, just return
        if restart and (len(get_keys(out_key)) > 0):
            return out_key

        # Fetch image data from S3
        image = get_s3_img(image_fname).astype(np.float32)
        if image.ndim == 2:
            image = image[None,...]
        elif image.ndim == 3:
            pass
        else:
            raise ValueError("Image shape must be (X,Y) or (Channels,X,Y)")

        image = np.maximum(image - darkframe, 0)

        # The second channel is flipped if it exists
        if image.shape[0] == 2:
            image[1,:,:] = image[1,:,::-1]

        # Main RL algorithm
        volume = rl.deconvolve(image, obj_xy, n_iters, kaiser_ord=kaiser_order)

        # Save to tiff stack
        put_s3_img(out_key, volume.astype(np.float32))


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('in_prefixes', type=str, nargs='+', help="Path(s) to frame tiffs")
    parser.add_argument('psf_prefix', type=str, help="Path(s) to PSF stack(s)")
    parser.add_argument('-d', '--darkframe', dest='darkframe_prefix', type=str,
                        default=None, help="Single dark frame (post-averaged)")
    parser.add_argument('-o', '--out-dir', dest='out_dir', type=str,
                        default='decon', help="Output directory")
    parser.add_argument('--obj-xy', dest='obj_xy', type=int, default=300,
                        help="Width of reconstructed object")
    parser.add_argument('-m', '--mags', dest='mags', type=float, default=[1.],
                        nargs='+', help="Relative magnifications of each lens "
                        "class")
    parser.add_argument('-n', '--n-iters', dest='n_iters', type=int, default=25,
                        help="Number of iterations of RL to perform")
    parser.add_argument('-p', '--padding', dest='padding', type=int, default=0,
                        help="Padding for convolutions")
    parser.add_argument('-k', '--kaiser-order', dest='kaiser_order', type=float, default=0.0,
                        help="Order of the Kaiser window to apply when rescaling volumes")
    parser.add_argument('-z', '--z-batch', dest='z_batch', type=int, default=8,
                        help="How many planes to FFT simultaneously (decrease if getting small OOMs)")
    parser.add_argument('-r', '--restart', dest='restart', action='store_true',
                        help="Skip if the output image already exists")
    args = parser.parse_args()

    main(**vars(args))
