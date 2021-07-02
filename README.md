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

There are 3 cloud-based jobs to configure which handle **deconvolution**, **PSF** preparation, and **movie** generation respectively. Public Docker containers are provided for each of these tasks, so recompiling these containers is not necessary.

###### AWS Batch - Compute environments

Using your AWS console, configure two compute environments as follows.

| Name         | Type    | Instance types | Min/Max/Desired CPU | On-demand price % |   |
|--------------|---------|----------------|---------------------|-------------------|---|
| r5-spot      | MANAGED | r5             | 0/256/0             | 100               |   |
| p3-16xl-spot | MANAGED | p3.16xlarge    | 0/4096/0            | 100               |   |

###### AWS Batch - Job queues



###### AWS Batch - Job definitions



## Workflow
