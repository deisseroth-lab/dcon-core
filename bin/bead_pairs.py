#!/usr/bin/env python
from os.path import basename, dirname, join
import numpy as np
import click
from natsort import natsorted
from joblib import Parallel, delayed
from s3tools import get_keys, get_s3_img, put_s3_img
from util import pos


@click.command()
@click.argument('img_prefix')
@click.option('-d', '--darkframe-path', type=str, default=None)
@click.option('-b', '--base-idx', type=int, default=0)
@click.option('-o', '--output-subdir', type=str, default="pairs")
def make_pairs(img_prefix, darkframe_path, base_idx, output_subdir):
    img_keys = natsorted(get_keys(img_prefix))
    print(f"{len(img_keys):d} images to process")

    baseframe_key = img_keys[base_idx]

    # Get the base frame and darkframe
    baseframe = get_s3_img(baseframe_key).astype(np.float32)
    
    if darkframe_path is None:
        darkframe = 100.
    else:
        darkframe = get_s3_img(darkframe_path).astype(np.float32)

    # Define function to be mapped
    def save_pair_img(img_key):
        img = get_s3_img(img_key).astype(np.float32)
        pair_img = pos(img + baseframe - 2 * darkframe)
        out_fname = join(dirname(img_key), output_subdir, basename(img_key))
        put_s3_img(out_fname, pair_img)
        return out_fname

    with Parallel(n_jobs=-1, verbose=4) as pool:
        output_fnames = pool(delayed(save_pair_img)(key) for key in img_keys)

    print(f"Finished processing {len(output_fnames):d} images")


if __name__ == "__main__":
    make_pairs()
