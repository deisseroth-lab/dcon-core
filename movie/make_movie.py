#!/usr/bin/env python
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu
import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter as VideoWriter
from moviepy.video.io.bindings import mplfig_to_npimage
from tqdm import tqdm
from s3tools import get_keys, get_s3_img, S3WriteBuffer

from functools import partial
from multiprocessing import Pool, cpu_count
from tempfile import NamedTemporaryFile

from util import crop_center
from viz import faces3d, plane_grid


DPI = 96
CHUNKSIZE = 8
N_PREVIEW = 25


def validate_types(types):
    if len(types) == 0:
        return MOVIE_TABLE.keys()
    else:
        return list(set(types) & set(MOVIE_TABLE.keys()))

def preprocess_volume(vol, crop_xy, crop_z):
    mid = [s // 2 for s in vol.shape[-3:]]
    c = [crop_z // 2, crop_xy // 2, crop_xy //2]
    return vol[mid[0]-c[0]:mid[0]+c[0],mid[1]-c[1]:mid[1]+c[1],mid[2]-c[2]:mid[2]+c[2]]

def identity_func(vol, dxy, dz, vmin, vmax):
    fig = faces3d(vol, dxy, dz, vmin=0, vmax=vmax, cmap='inferno')
    return fig

def identity_txfm(vols, dxy, dz):
    low_percentile = 25
    high_percentile = 99.999

    # Get zero-value
    low_vol = np.min(vols, axis=0)
    th = threshold_otsu(low_vol)
    fg = low_vol[low_vol > th]
    vmin = np.percentile(fg, low_percentile)

    # Get max-value
    high_vol = np.max(vols, axis=0)
    fg = high_vol[low_vol > th]
    vmax = np.percentile(fg, high_percentile)

    return partial(identity_func, dxy=dxy, dz=dz, vmin=vmin, vmax=vmax)

def dff_func(vol, f_vol, mask, dxy, dz, dff_max):
    dff_vol = (vol - f_vol) / f_vol
    cmap = sns.diverging_palette(240, 12, s=80, l=55, center='dark', as_cmap=True)
    fig = faces3d(dff_vol * mask, dxy, dz, vmin=-dff_max, vmax=dff_max, cmap=cmap)
    return fig

def dff_txfm(vols, dxy, dz):
    low_percentile = 5
    high_percentile = 99.999
    dilation_iters = 3

    f_vol = np.median(vols, axis=0)

    # Mask out background
    th = threshold_otsu(f_vol)
    fg = f_vol[f_vol > th]
    mask = f_vol > np.percentile(fg, low_percentile)
    mask = binary_dilation(mask, iterations=dilation_iters)

    vmax = np.percentile(abs(vols - f_vol) * mask / f_vol, high_percentile)

    return partial(dff_func, f_vol=f_vol, mask=mask, dxy=dxy, dz=dz, dff_max=vmax)

def planes_func(vol, dxy, dz, vmin, vmax):
    fig = plane_grid(vol, (3, 3), dxy, dz, vmin=vmin, vmax=vmax)
    return fig

def planes_txfm(vols, dxy, dz):
    low_percentile = 25
    high_percentile = 99.999

    # Get zero-value
    low_vol = np.min(vols, axis=0)
    th = threshold_otsu(low_vol)
    fg = low_vol[low_vol > th]
    vmin = np.percentile(fg, low_percentile)

    # Get max-value
    high_vol = np.max(vols, axis=0)
    fg = high_vol[low_vol > th]
    vmax = np.percentile(fg, high_percentile)

    return partial(planes_func, dxy=dxy, dz=dz, vmin=vmin, vmax=vmax)

def dff_planes_func(vol, f_vol, mask, dxy, dz, dff_max):
    dff_vol = (vol - f_vol) / f_vol
    cmap = sns.diverging_palette(240, 12, s=80, l=55, center='dark', as_cmap=True)
    fig = plane_grid(dff_vol * mask, (3, 3), dxy, dz, vmin=-dff_max, vmax=dff_max, cmap=cmap)
    return fig

def dff_planes_txfm(vols, dxy, dz):
    low_percentile = 5
    high_percentile = 99.999
    dilation_iters = 3

    f_vol = np.median(vols, axis=0)

    # Mask out background
    th = threshold_otsu(f_vol)
    fg = f_vol[f_vol > th]
    mask = f_vol > np.percentile(fg, low_percentile)
    mask = binary_dilation(mask, iterations=dilation_iters)

    vmax = np.percentile(abs(vols - f_vol) * mask / f_vol, high_percentile)

    return partial(dff_planes_func, f_vol=f_vol, mask=mask, dxy=dxy, dz=dz, dff_max=vmax)

def make_frame(key, txfms, figsize, crop_xy, crop_z):
    plt.switch_backend('agg')
    plt.style.use('dark_lab')
    vol = get_s3_img(key)
    vol = preprocess_volume(vol, crop_xy, crop_z)
    arrs = []
    for txfm in txfms:
        fig = plt.figure(figsize=figsize, dpi=DPI)
        fig = txfm(vol)
        fig.set_facecolor('k')
        arrs.append(mplfig_to_npimage(fig))
        plt.close(fig)
    return arrs


MOVIE_TABLE = {
    "identity": identity_txfm,
    "dff": dff_txfm,
    "planes": planes_txfm,
    "dff_planes": dff_planes_txfm
}


@click.command()
@click.argument('img_prefix', type=str)
@click.option('-t', '--type', 'movie_types', multiple=True)
@click.option('-f', '--fs', 'fs', type=float, default=20.0)
@click.option('--dxy', 'dxy', type=float, default=1.121)
@click.option('--dz', 'dz', type=float, default=1.5)
@click.option('-s', '--imsize', 'imsize', type=float, default=720)
@click.option('--crop-xy', 'crop_xy', type=int, default=512)
@click.option('--crop-z', 'crop_z', type=int, default=200)
@click.option('-p', '--preset', 'preset', type=str, default='medium')
@click.option('--crf', 'crf', type=int, default=16)
def make_movie(img_prefix, movie_types, fs, dxy, dz, imsize, crop_xy, crop_z, preset, crf):
    movie_types = validate_types(movie_types)
    print("Producing image types: " + str(movie_types))

    img_keys = get_keys(img_prefix)
    print("{:d} frames detected".format(len(img_keys)))

    figsize = (imsize / DPI, imsize / DPI)

    with Pool(cpu_count()) as pool:
        test_indices = np.linspace(0, len(img_keys), num=N_PREVIEW, endpoint=False).astype(np.int)
        test_volumes = list(tqdm(pool.imap(get_s3_img, [img_keys[idx] for idx in test_indices]), total=N_PREVIEW, desc="Preview"))
        test_volumes = [preprocess_volume(v, crop_xy, crop_z) for v in test_volumes]
        test_volumes = np.stack(test_volumes)

        transforms = [MOVIE_TABLE[k](test_volumes, dxy, dz) for k in movie_types]

        make_frame_specific = partial(make_frame, txfms=transforms, figsize=figsize, crop_xy=crop_xy, crop_z=crop_z)
        frames = tqdm(pool.imap(make_frame_specific, img_keys, chunksize=CHUNKSIZE), total=len(img_keys), desc="Frames")

        video_files = [NamedTemporaryFile(suffix='.mp4') for _ in movie_types]
        video_writers = [VideoWriter(video_file.name,
                                     size=(imsize, imsize),
                                     fps=fs,
                                     codec='libx264',
                                     preset=preset,
                                     ffmpeg_params=['-crf', '{:d}'.format(crf)])
                         for video_file in video_files]

        for frame in frames:
            for idx, vid_writer in enumerate(video_writers):
                vid_writer.write_frame(frame[idx])

        for vid_writer, video_file, movie_type in zip(video_writers, video_files, movie_types):
            vid_writer.close()

            movie_basename = 'movie_{:s}.mp4'.format(movie_type)
            out_fname = '/'.join(img_keys[0].split('/')[:-1] + [movie_basename])
            print("Saving as " + out_fname)

            video_file.seek(0)
            with S3WriteBuffer(out_fname, {'ContentType': 'video/mp4'}) as s3wb:
                s3wb.write(video_file.read())

            video_file.close()

if __name__ == '__main__':
    make_movie()
