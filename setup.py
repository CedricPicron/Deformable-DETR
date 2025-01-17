"""
Builds Deformable DETR package.
"""

import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "models/ops/src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.2;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "deformable_detr.models.ops.MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


packages = find_packages(exclude=("*datasets", "models.ops"))
packages = [f'deformable_detr.{package}' for package in packages]
package_dir = {'deformable_detr': ''}

setup(
    package_dir=package_dir,
    packages=packages,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.5.1",
        "torchvision>=0.6.1",
        "pycocotools>=2.0.2",
        "tqdm>4.29.0",
        "cython",
        "scipy"
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
