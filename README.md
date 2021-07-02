# Dcon Core

Tools for reconstructing 3D volumes from aperture-plane light-field images

## Features

* Super-sampling reconstruction
* GPU-based Fourier convolution
* Efficient data handling for multi-GPU systems
* Utilities for preparing empirical point spread functions (PSFs) and generating volume movies
* Tools for running everything as batch jobs on the AWS cloud

## Requirements

* A Unix machine such as Linux, macOS, or WSL
* At least one GPU and enough total GPU memory to store your PSF stack at 32-bit resolution in GPU memory
* PyTorch and other common scientific Python libraries

## Installation

#### Overview

This package consists of several small pre-/post-processing scripts which are to be run locally, and several workhorse job scripts such as `deconvolve/decon_local.py` which are to be run on a GPU-capable machine or on AWS. When run in the cloud, local `submit_` scripts are used to submit the jobs to AWS.

Docker is used to build unique containers for each of the workhorse jobs. However docker local installation is currently not supported - we use Anaconda or virtualenvs for that instead.

#### Local

###### Pip

Create a virtual environment (preferably) then

```
git clone git@github.com:deisseroth-lab/dcon-core.git
cd dcon-core
pip install -e .
```

###### Anaconda

```
git clone git@github.com:deisseroth-lab/dcon-core.git
cd dcon-core
conda env create --file environment.yml
```

#### AWS configuration

There are 3 cloud-based jobs to configure which handle **deconvolution**, **PSF** preparation, and **movie** generation respectively. Public Docker containers are provided for each of these tasks, so recompiling these containers is not necessary. Below is an example configuration which we use in our production setup.

###### AWS Batch - Compute environments

Using your AWS console, configure two compute environments as follows.

| Name         | Type    | Instance types | Min/Max/Desired CPU | On-demand price % |
|--------------|---------|----------------|---------------------|-------------------|
| r5-spot      | MANAGED | r5             | 0/256/0             | 100               |
| p3-16xl-spot | MANAGED | p3.16xlarge    | 0/4096/0            | 100               |

###### AWS Batch - Job queues

Configure the following job queues.

| Name            | Compute environment |
|-----------------|---------------------|
| dcon-psfs       | r5-spot             |
| dcon-deconvolve | p3-16xl-spot        |
| dcon-movie      | r5-spot             |

###### AWS Batch - Job definitions

Register the following job definitions. `Command`, `Timeout`, `vCPUs`, `Memory`, and other critical parameters are overritten by the job submission scripts and do not need to be specified correctly here.

| Name            | Platform | Attempts     | Image                                   | Command syntax |
|-----------------|----------|--------------|-----------------------------------------|----------------|
| dcon-psfs       | EC2      | r5-spot      | public.ecr.aws/d1v4a7q2/dcon-psfs       | Bash           |
| dcon-deconvolve | EC2      | p3-16xl-spot | public.ecr.aws/d1v4a7q2/dcon-deconvolve | Bash           |
| dcon-movie      | EC2      | r5-spot      | public.ecr.aws/d1v4a7q2/dcon-movie      | Bash           |

## Workflow

Image acquisition > local preprocessing > **PSF preparation** > **Volume reconstruction** > **Movie rendering**

All steps in **bold** require inputs to be available on AWS S3 and will not accept locally-stored data. Typically you will want to run them using AWS Batch or from the command line of an appropriate EC2 instance. It is not efficient to run these locally due to the data transfer to and from S3!

### Acquisition and preprocessing

#### Camera configuration

The software expects one or two cameras. If two, then it is assumed that the first is looking straight through a 90-degree beam-splitter and the second camera looking from the side and seeing an image mirrored about the vertical axis.

#### Using MicroManager

We used MicroManager's Multidimensional Acquisition manager to perform several data collection tasks.

###### PSF acquisition

Perform 4 Z-stack acquisitions at a 2x2 grid of XY locations. These are saved as a series of 4 TIFF stacks. A second acquisition is performed in a dark area of the sample, providing a "dark stack" used for background subtraction.

###### Z-sweeps and time-series

For these experiments we *save individual TIFF files* as opposed to stacks. After each acquisition, we used `bin/clean_mm_paths.py` to rename and restructure data for later processing. `bin/fix_z_shifts.py` prepares data from Z-sweeping experiments while `bin/fix_multichannel.py` prepares data from time-series experiments, such as live subject imaging.

#### Using DCImage Live (Hamamatsu)

While convenient to use, MicroManager is unable to manage acquisition from our high-speed cameras at their maximum frame rate. Specifically, MicroManager is unable to write frames to disk fast enough to keep the in-memory buffer from filling, which for us took a matter of seconds with 32GB of buffer. This is the case even when the underlying hardware (4-way NVMe SSD RAID) is fast enough to support the required data rate.

Instead we use a small custom LabView script to configure and perform time series imaging experiments with external pulse synchronization. To write the data to disk efficiently, Hamamatsu cameras utilize a proprietary "DCImage" file format, generating one file per camera. We used `bin/dcimg_to_tiff` to extract individual TIFFs post-hoc from DCImage files.

#### Darkframes

It is helpful to perform a "dark frame" acquisition to aid in background subtraction, similar to the "dark stacks" used with PSF acquisition. For each set of imaging parameters (exposure, illumination power, imaging medium) we acquired 25 frames with no subject in view, usually by commanding the stage a few millimeters to the side. We generate appropriate TIFF files as described above, then average these together using `bin/make_darkframe` to create a single light field image used for background subtraction.

### PSF processing

```
# Submit to Batch
bin/submit_psfs --help

# Run directly
psfs/clean_psfs.py --help
```

###### Centers file

The PSF preparation script needs to know which lenslet images correspond to which focal depths. This information is contained in a "centers file", which is a tab-separated CSV of the following form.

| y | x | c |
|---|---|---|
|123|654|  0|

Where each row lists the approximate vertical and horizontal position of a lenslet image center on the sensor followed by the lenslet class (0 or 1 in our case). The same centers file is applied to both cameras, accounting for the mirroring in the image splitter automatically.

### Deconvolution

```
# Submit to Batch
bin/submit_decon --help

# Run directly
deconvolve/decon_local.py --help
```

### Volume-movie rendering

```
# Submit to Batch
bin/submit_movie --help

# Run directly
movie/make_movie.py --help
```
