"""
Builds Deformable DETR package.
"""

from setuptools import find_packages, setup


setup(
    packages=find_packages(exclude=("configs", "datasets", "docs", "figs")),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.5.1",
        "torchvision>=0.6.1",
        "pycocotools>=2.0.2",
        "tqdm>4.29.0",
        "cython",
        "scipy"
    ]
)
