from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Turing (T4) for free-tier development, Ampere (A100, A10), Ada (L4), Hopper (H100).
ARCHS = ["75", "80", "86", "89", "90"]
gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in ARCHS]

setup(
    name="cuda_attn",
    version="0.1.0",
    description="Naive, tiled and fused (online-softmax) CUDA attention kernels",
    packages=find_packages(exclude=("tests", "bench")),
    ext_modules=[
        CUDAExtension(
            name="cuda_attn._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/attn_naive.cu",
                "csrc/attn_tiled.cu",
                "csrc/attn_fused.cu",
                "csrc/attn_fused_tuned.cu",
                "csrc/attn_fused_tc.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                # -lineinfo lets Nsight Compute map hot spots back to source lines.
                "nvcc": ["-O3", "-lineinfo", "--use_fast_math", *gencode],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
