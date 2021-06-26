#!/usr/bin/env python
import numpy as np
import pandas as pd
import click
from joblib import Parallel, delayed
import re
from os.path import dirname
from s3tools import get_keys, get_s3_img, put_s3_img
from dcon.util import pos

MM_TIF_REGEX = r"img_channel(?P<ch>\d{3})_position(?P<pos>\d{3})_time(?P<t>\d{9})_z(?P<z>\d{3})\.tif"

@click.command()
@click.argument('img_prefix')
def fix_multichannel(img_prefix):
    """When you save as separate tiff files to get time separated out
    in a time series experiment, it also splits the channels. Fix that with this.
    """
    img_keys = get_keys(img_prefix)
    print(f"{len(img_keys):d} images to process")

    p = re.compile(MM_TIF_REGEX)
    key_info_df = pd.DataFrame(map(lambda key: p.search(key).groupdict(), img_keys))

    # Determine which fields are being used
    for col_name in key_info_df.columns:
        if key_info_df[col_name].nunique() < 2:
            key_info_df.drop(columns=col_name, inplace=True)

    non_channel_cols = key_info_df.columns.drop('ch')

    channel_info_groups = key_info_df.groupby(non_channel_cols.to_list())

    def combine_channels(names, group):
        new_key = "img"
        for col_name, name in zip(non_channel_cols, names):
            new_key += "_" + col_name + name
        new_key += ".tif"
        indices = sorted(group.index)
        out_img = np.stack([get_s3_img(img_keys[idx]) for idx in indices], axis=0)
        out_fname = dirname(img_keys[indices[0]]) + "/" + new_key
        put_s3_img(out_fname, out_img)
        return out_fname

    with Parallel(n_jobs=-1, verbose=4) as pool:
        output_fnames = pool(delayed(combine_channels)(names, group) for (names, group) in channel_info_groups)

    print(f"Finished processing {len(output_fnames):d} images")


if __name__ == "__main__":
    fix_multichannel()
