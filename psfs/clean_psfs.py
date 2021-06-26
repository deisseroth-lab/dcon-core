#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from scipy.spatial.distance import pdist
from skimage.measure import label, ransac
from skimage.morphology import disk, binary_dilation
from skimage.transform import ProjectiveTransform, AffineTransform

import logging
from os.path import expanduser, dirname, join
import warnings

from s3tools import get_s3_img, put_s3_img, get_keys, s3open, S3WriteBuffer
from dcon.util import pos, crop_around, max_inds, reshape_for_channels, Timer
from dcon.viz import faces3d


CAM_OFFSET = 100.0
DXY, DZ = 1.121, 1.5
CH_OFFSET_UM = 250 / 4


def get_constrained_centers(plane, guess_centers, centers_map, search_radius, txfm_type=AffineTransform):
    mask = disk(search_radius)
    bright_centers = np.zeros_like(guess_centers)
    for idx, guess in enumerate(guess_centers):
        guess_coord = np.round(guess).astype(int)
        offset = guess_coord - search_radius
        wide_roi = crop_around(plane, guess_coord, search_radius, include_center=True)
        bright_center = np.array(max_inds(wide_roi * mask)) + offset
        # Deal with failure to find a bright spot
        if plane[bright_center[0],bright_center[1]] == 0:
            bright_center = guess_coord
        bright_centers[idx] = bright_center

    txfm, inliers = ransac((centers_map, bright_centers), txfm_type, min_samples=3, residual_threshold=3, max_trials=100)
    #txfm = AffineTransform()
    #txfm.estimate(centers_map, bright_centers)
    constrained_centers = txfm(centers_map)
    return constrained_centers

def clean_roi(roi, thresh=0, max_offcenter=64):
    X, Y = roi.shape
    mask = disk(X//2).astype(np.bool)

    mu = roi[mask].mean()
    sd = roi[mask].std()

    fg_mask = (roi >= mu + thresh * sd)

    label_image, N = label(fg_mask, return_num=True)

    near_mask = np.zeros_like(fg_mask)
    near_mask[X//2,Y//2] = 1
    near_mask = binary_dilation(near_mask, disk(max_offcenter))

    masked_labels = near_mask * label_image
    max_coords = max_inds(masked_labels)

    max_label = label_image[max_coords]
    # Deal with failure to find a connected region
    if max_label == 0:
        spot_mask = np.zeros_like(roi)
    else:
        spot_mask = label_image == max_label
    return roi * spot_mask

def clean_plane(plane, centers, classes, roi_radius):
    X, Y = plane.shape
    C = len(np.unique(classes))
    offsets = centers.astype(np.int) - roi_radius
    rois = np.stack([crop_around(plane, center.astype(np.int), roi_radius, True) for center in centers])
    cleaned_plane = np.zeros((C, X, Y), dtype=plane.dtype)
    circle_mask = disk(roi_radius).astype(np.bool)
    for roi, offset, clas in zip(rois, offsets, classes):
        # Just use the raw ROI!
        cleaned_plane[clas, offset[0]:offset[0]+2*roi_radius+1,offset[1]:offset[1]+2*roi_radius+1] = roi * circle_mask
    return cleaned_plane

def put_s3_fig(path):
    with S3WriteBuffer(path, {'ContentType': 'image/png'}) as s3wb:
        plt.savefig(s3wb)

def brightness_profiles(ss_stacks, dark_stack, out_dir):
    """Check the brightness profile of the raw stacks"""
    S = len(ss_stacks)
    Ch = ss_stacks[0].shape[0]
    out_path = join(out_dir, "brightness_profiles.png")
    fig, ax = plt.subplots(1, Ch, figsize=(12, 4))
    for ch in range(Ch):
        for s in range(S):
            ax[ch].plot(ss_stacks[s][ch].mean(axis=(-2,-1)), label="Supersample {:d}".format(s))
        ax[ch].plot(dark_stack[ch].mean(axis=(-2,-1)), lw=3, ls='--', label="Darkstack")
        ax[ch].set_title("Channel {:d}".format(ch))
        ax[ch].set_xlabel("Plane number")
        ax[ch].legend()
    ax[0].set_ylabel("Raw pixel intensity")
    put_s3_fig(out_path)

def subbed_stacks(subbed, out_dir):
    S = len(subbed)
    Ch = subbed[0].shape[0]
    out_path = join(out_dir, "subbed_profiles.png")
    fig, ax = plt.subplots(1, Ch, figsize=(12, 4))
    for ch in range(Ch):
        for s in range(S):
            ax[ch].plot(subbed[s][ch].mean(axis=(-2,-1)), label="Supersample {:d}".format(s))
        ax[ch].set_title("Channel {:d}".format(ch))
        ax[ch].set_xlabel("Plane number")
        ax[ch].legend()
    ax[0].set_ylabel("Subtracted pixel intensity")
    put_s3_fig(out_path)

def plot_interlaced(interlaced, out_dir):
    plt.figure(figsize=(20, 20))
    out_path = join(out_dir, "interlaced.png")
    faces3d(interlaced.sum(axis=0), cmap='inferno', vmin=0)
    put_s3_fig(out_path)

def plot_centers_linear(centers_history, out_dir):
    Ch = centers_history.shape[0]
    out_path = join(out_dir, "centers_linear.png")
    fig, ax = plt.subplots(2, Ch, figsize=(6 * Ch, 8))
    for ch in range(Ch):
        ax[0,ch].plot(centers_history[ch,:,:,0])
        ax[0,ch].set_title("Channel {:d}".format(ch))
        ax[0,ch].set_ylabel("Row coord")
        ax[1,ch].plot(centers_history[ch,:,:,1])
        ax[1,ch].set_title("Channel {:d}".format(ch))
        ax[1,ch].set_ylabel("Col coord")
    put_s3_fig(out_path)

def plot_centers_2d(centers_history, interlaced_shape, out_dir):
    Ch = centers_history.shape[0]
    out_path = join(out_dir, "centers_2d.png")
    fig, ax = plt.subplots(1, Ch, figsize=(20, 10))
    for ch in range(Ch):
        ax[ch].plot(centers_history[ch,:,:,1], centers_history[ch,:,:,0])
        ax[ch].set_xlim([0, interlaced_shape[-1]])
        ax[ch].set_ylim([interlaced_shape[-2], 0])
        ax[ch].set_aspect('equal')
    put_s3_fig(out_path)

def plot_class_stacks(stacks, out_dir):
    out_path = join(out_dir, "class_stacks.png")
    plt.figure(figsize=(20, 20))
    faces3d(stacks.sum(axis=(0, 1)), vmin=0, cmap='inferno')
    put_s3_fig(out_path)
    
def plot_z_distribution(stacks, dxy, dz, out_dir):
    out_path = join(out_dir, "z_distribution.png")
    colors = ['c', 'g', 'm', 'r']
    aspect = 3
    pal = sns.color_palette(colors)
    Ch, C, Z, X, Y = stacks.shape
    z_mid_um = Z * dz / 2
    
    projections = stacks.max(axis=-2)[:,:,::-1,:]
    cdx = np.arange(Ch * C).reshape((Ch, C))
    cm = np.zeros((Z, Y, 3))
    for ch in range(Ch)[::-1]:
        for c in range(C)[::-1]:
            color = pal[cdx[ch,c]]
            psf = projections[ch,c]
            cm += np.stack([psf * c for c in color], axis=-1)
    
    scale = np.max(cm)

    fig, ax = plt.subplots(1, 1, figsize=(20, aspect * 2.5))
    ax.imshow(cm / scale, extent=[0, Y*dxy, 0, Z * dz], aspect=aspect)
    ax.set_ylabel("Depth (um)")
    ax.axhline(z_mid_um + CH_OFFSET_UM, ls='--', c='gray')
    ax.axhline(z_mid_um - CH_OFFSET_UM, ls='--', c='gray')
    put_s3_fig(out_path)

    
@click.command()
@click.argument('psf_prefix')
@click.argument('dark_prefix', default=None)
@click.argument('centers_key', default="s3://xlfm-dcon/S20_centers.csv")
@click.option('-o', '--out-dir', 'out_dir', type=str, default="")
@click.option('-c', '--channels', 'channels', type=int, default=1)
@click.option('-r', '--rel-rad', 'rel_rad', type=float, default=1.0)
def clean_psfs(psf_prefix, dark_prefix, centers_key, out_dir, channels, rel_rad):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    full_out_dir = join(dirname(psf_prefix), out_dir)
    plt.style.use('dark_lab')

    with Timer("Loading stacks"):
        # Load PSFs
        psf_keys = [k for k in get_keys(psf_prefix) if "bleach" not in k]
        ss_psf_stacks = []
        for key in tqdm(psf_keys, desc="Loading PSFs"):
            psf = get_s3_img(key)
            psf = reshape_for_channels(psf, channels)
            psf = psf.astype(np.float32)
            ss_psf_stacks.append(psf)

        S = len(ss_psf_stacks)
        Ch, Z, X, Y = ss_psf_stacks[0].shape

        # Load dark frame
        if dark_prefix:
            dark_keys = get_keys(dark_prefix)
            N_dark = len(dark_keys)
            dark_stack = np.zeros((N_dark, Ch, Z, X, Y), dtype=np.uint16)
            for idx, key in tqdm(enumerate(dark_keys), total=len(dark_keys), desc="Loading darkframes"):
                dark = get_s3_img(key)
                dark = reshape_for_channels(dark, channels)
                dark_stack[idx] = dark
            del dark
            dark_stack = np.median(dark_stack, axis=0).astype(np.float32)
        else:
            dark_stack = CAM_OFFSET

        brightness_profiles(ss_psf_stacks, dark_stack, full_out_dir)

    with Timer("Subtracting and interlacing"):
        # Dark frame subtraction
        for s in trange(S, desc="Subtracting"):
            ss_psf_stacks[s] = pos(ss_psf_stacks[s] - dark_stack)
        del dark_stack

        subbed_stacks(ss_psf_stacks, full_out_dir)

        # Super sample interlacing
        interlaced = np.zeros((Ch, Z, X*S//2, Y*S//2))
        X, Y = interlaced.shape[-2:]
        sel = np.arange(S).reshape((S//2, S//2))
        for sx in range(S//2):
            for sy in range(S//2):
                interlaced[:,:,sx::2,sy::2] = ss_psf_stacks[sel[sx,sy]]
        del ss_psf_stacks

        plot_interlaced(interlaced, full_out_dir)

    with Timer("Loading and tracking lenslet centers"):
        # Load centers
        dsf = 2048 / X
        with s3open(centers_key, 'rb') as centers_buffer:
            centers_df = pd.read_csv(centers_buffer, sep='\t')
        lenses = len(centers_df)
        C = len(centers_df['c'].unique())
        centers = centers_df[['y', 'x']].values / dsf
        classes = centers_df['c'].values

        # Fit and track lenslet centers through the planes
        search_radius = int(pdist(centers).min() * 0.5 * rel_rad)
        bead_mid_offset = 0

        class_stack = np.zeros((C, Ch, Z, X, Y), dtype=np.float32)
        centers_history = np.zeros((Ch, Z, len(centers), 2), dtype=np.float32)

        for ch in range(Ch):
            stack = interlaced[ch]

            # Find in mid then
            # Search up
            init_plane_idx = int(Z * (1 + 2 * ch) / (2 * Ch)) + bead_mid_offset
            plane_centers = centers
            for plane_idx in trange(init_plane_idx, Z, desc="Upward pass"):
                plane = stack[plane_idx]
                if plane_idx == init_plane_idx:
                    plane_centers = get_constrained_centers(plane, plane_centers, centers, search_radius)
                else:
                    plane_centers = get_constrained_centers(plane, plane_centers, plane_centers, 10)
                cleaned_plane = clean_plane(plane, plane_centers.astype(np.int), classes, search_radius)
                class_stack[:,ch,plane_idx] = cleaned_plane
                centers_history[ch,plane_idx] = plane_centers

            # Search down
            init_plane_idx = int(Z * (1 + 2 * ch) / (2 * Ch)) - 1 + bead_mid_offset
            plane_centers = centers
            for plane_idx in trange(init_plane_idx, -1, -1, desc="Downward pass"):
                plane = stack[plane_idx]
                if plane_idx == init_plane_idx:
                    plane_centers = get_constrained_centers(plane, plane_centers, centers, search_radius)
                else:
                    plane_centers = get_constrained_centers(plane, plane_centers, plane_centers, 10)
                cleaned_plane = clean_plane(plane, plane_centers.astype(np.int), classes, search_radius)
                class_stack[:,ch,plane_idx] = cleaned_plane
                centers_history[ch,plane_idx] = plane_centers

        plot_centers_linear(centers_history, full_out_dir)
        plot_centers_2d(centers_history, interlaced.shape, full_out_dir)

    with Timer("Norming"):
        # Norming
        plane_sums = class_stack.sum(axis=(0, 1, 3, 4))
        class_stack = class_stack / plane_sums[...,None,None]

        plot_class_stacks(class_stack, full_out_dir)
        plot_z_distribution(class_stack, DXY, DZ, full_out_dir)

    with Timer("Saving"):
        metac = 0
        for ch in trange(Ch, leave=False, desc="Channel"):
            for c in trange(C, leave=False, desc="Class"):
                out_file = join(full_out_dir, f"cleaned_{metac:d}.tif")
                tqdm.write(f"Saving interlaced PSF group {metac:d} to {out_file:s}")
                put_s3_img(out_file, class_stack[c,ch,...])
                metac += 1


if __name__ == '__main__':
    clean_psfs()
