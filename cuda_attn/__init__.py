"""Hand-written CUDA attention kernels.

    import cuda_attn
    o = cuda_attn.fused(q, k, v, causal=False)   # q, k, v: (B, H, N, d) CUDA tensors

If the package was installed with `pip install -e .` the prebuilt extension is used.
Otherwise the kernels are JIT-compiled on the first call (takes about a minute).
"""
from __future__ import annotations

import os

import torch

_EXT = None  # cached extension module (not named _C: that would shadow the compiled submodule)


def _ext():
    """Load the compiled extension on first use."""
    global _EXT
    if _EXT is None:
        try:
            from . import _C as ext  # built by setup.py
        except ImportError:
            from torch.utils.cpp_extension import load

            root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc")
            ext = load(
                name="cuda_attn_jit",
                sources=[os.path.join(root, f) for f in
                         ("bindings.cpp", "attn_naive.cu", "attn_tiled.cu", "attn_fused.cu",
                          "attn_fused_tuned.cu", "attn_fused_tc.cu")],
                extra_include_paths=[root],
                extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
            )
        _EXT = ext
    return _EXT


def naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """v1: three kernels, one thread per output, N x N scores in HBM."""
    return _ext().naive(q, k, v, causal)


def tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """v2: shared-memory tiled matmuls and a warp-per-row softmax, N x N scores still in HBM."""
    return _ext().tiled(q, k, v, causal)


def fused(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """v3: one fused kernel with online softmax; scores never leave the SM. Requires d == 64."""
    return _ext().fused(q, k, v, causal)


def fused_tuned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
                double_buffer: bool = True, br: int = 64, bc: int = 32) -> torch.Tensor:
    """v3.1 (double_buffer=False): 128-bit loads, FP16 tiles in shared memory.
    v3.2 (double_buffer=True): plus cp.async prefetch of the next K/V tile.
    (br, bc) in {(64, 32), (64, 64), (128, 32), (128, 64)}. FP16, d == 64."""
    return _ext().fused_tuned(q, k, v, causal, double_buffer, br, bc)


def fused_vec(q, k, v, causal=False):
    """v3.1 with default tiles."""
    return fused_tuned(q, k, v, causal, double_buffer=False)


def fused_db(q, k, v, causal=False):
    """v3.2 with default tiles."""
    return fused_tuned(q, k, v, causal, double_buffer=True)


def fused_tc(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """v4: both matmuls on tensor cores (WMMA), online softmax in between. FP16, d == 64, sm_70+."""
    return _ext().fused_tc(q, k, v, causal)


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """FP32 math-backend SDPA: the ground truth every kernel is tested against."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=causal)


# Every version, in the order the project builds them.
KERNELS = {"naive": naive, "tiled": tiled, "fused": fused,
           "fused_vec": fused_vec, "fused_db": fused_db, "fused_tc": fused_tc}
# Versions that only take float16 inputs.
FP16_ONLY = {"fused_vec", "fused_db", "fused_tc"}

__all__ = ["naive", "tiled", "fused", "fused_tuned", "fused_vec", "fused_db", "fused_tc",
           "reference", "KERNELS", "FP16_ONLY"]
