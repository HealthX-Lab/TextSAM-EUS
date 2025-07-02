# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

from setuptools import find_packages, setup

setup(
    name="TextSAM_EUS",
    version="0.0.1",
    author="Taha Koleilat",
    python_requires=">=3.9",
    install_requires=["monai", "open_clip_torch", "pandas", "matplotlib", "easydict", "scikit-image", "albumentations", "SimpleITK>=2.2.1", "ftfy", "transformers", "nibabel", "tqdm", "scipy", "ipympl", "connected-components-3d", "opencv-python", "jupyterlab", "ipywidgets"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
