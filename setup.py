#!/usr/bin/env python
from setuptools import setup

setup(
    name="dcon",
    description="Deconvolve aperture plane-space light fields",
    license="MIT",
    author="Noah Young",
    author_email="npyoung@stanford.edu",
    url="https://github.com/deisseroth-lab/dcon-core",
    packages=['dcon'],
    install_requires=[
        'click',
        'numpy==1.17.2',
        'matplotlib==3.1.1',
        'matplotlib-scalebar',
        'scipy==1.3.1',
        'joblib==0.13.2',
        'tqdm',
        'natsort',
        'tifffile',
        'boto3==1.9.246',
        's3tools @ git+https://github.com/npyoung/s3tools'
    ],
    extras_require={
        'deconvolve': [
            'torch==1.5.0',
        ],
        'psfs': [
            'pandas==0.25.1',
            'seaborn==0.8.1',
            'scikit-image==0.15.0',
        ],
        'movie': [
            'moviepy==1.0.0',
            'scikit-image==0.15.0',
            'seaborn==0.10.0'
        ]
    },
    scripts=[
        'bin/nvmon',
        'bin/dcimg_to_tiff',
        'bin/make_darkframe',
        'bin/submit_psfs',
        'bin/submit_decon',
        'bin/submit_movie',
        'bin/clear_batch_queue'
    ]
)
