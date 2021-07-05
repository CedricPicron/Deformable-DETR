"""
Builds Deformable DETR package.
"""

from setuptools import setup


setup(
    package_dir={'deformable-detr.models': "models"},
    packages=["deformable-detr.models"],
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
