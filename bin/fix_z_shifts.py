#!/usr/bin/env python
import numpy as np
import click
from natsort import natsorted
from joblib import Parallel, delayed
from s3tools import get_keys, get_s3_img, put_s3_img
from dcon.util import pos


@click.command()
@click.argument('img_prefix')
def fix_z_shifts(img_prefix):
    """When you save as separate tiff files to get the planes separated out
    in a z_shifts experiment, it also splits the channels. Fix that with this.
    """
    img_keys = natsorted(get_keys(img_prefix))
    print(f"{len(img_keys):d} images to process")

    halfway = len(img_keys) // 2
    ch0_keys = img_keys[:halfway]
    ch1_keys = img_keys[halfway:]

    # Define function to be mapped
    def combine_channels(ch0, ch1):
        out_img = np.stack([get_s3_img(ch0), get_s3_img(ch1)], axis=0)
        parts = ch0.rsplit("/", 1)
        out_fname = parts[0] + "/" + parts[1][-8:]
        put_s3_img(out_fname, out_img)
        return out_fname

    with Parallel(n_jobs=-1, verbose=4) as pool:
        output_fnames = pool(delayed(combine_channels)(ch0, ch1) for (ch0, ch1) in zip(ch0_keys, ch1_keys))

    print(f"Finished processing {len(output_fnames):d} images")


if __name__ == "__main__":
    fix_z_shifts()
