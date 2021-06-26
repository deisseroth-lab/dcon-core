#!/usr/bin/env python
import click
from glob import glob
from os import listdir, rename, remove, rmdir
from os.path import join, dirname, basename


@click.command()
@click.argument('experiment_path_pattern', type=str)
def main(experiment_path_pattern):
    experiment_roots = glob(experiment_path_pattern)
    for expt_root in experiment_roots:
        # Find all the tifs and metadata.txt
        tifs = glob(join(expt_root, "**", "*.tif"), recursive=True)
        meta = glob(join(expt_root, "**", "metadata.txt"), recursive=True)

        # Move tifs and meta
        for tif in tifs:
            rename(tif, join(expt_root, basename(tif)))
        for m in meta:
            rename(m, join(expt_root, basename(m)))

        # Remove display settings
        try:
            remove(join(expt_root, "DisplaySettings.json"))
        except FileNotFoundError:
            print("No DisplaySettings.json file found")

        # Remove remaining subdirs
        subdirs = glob(join(expt_root, "*/"))
        for subdir in subdirs:
            rmdir(subdir)

    return 0

if __name__ == '__main__':
    main()