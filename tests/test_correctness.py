"""GPU correctness suite: every kernel vs FP32 math-backend SDPA.

Skipped automatically on machines without CUDA.
"""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

import cuda_attn  # noqa: E402

TOL = {torch.float32: 1e-4, torch.float16: 2e-3}
SHAPES_N = [1, 15, 16, 63, 64, 65, 128, 512, 1000, 1024, 2048]
ALL = list(cuda_attn.KERNELS)


def make_qkv(b, h, n, d, dtype, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [(torch.randn(b, h, n, d, device="cuda", dtype=torch.float32, generator=g) * scale).to(dtype)
            for _ in range(3)]


def assert_close(out, ref, dtype):
    err = (out.float() - ref.float()).abs().max().item()
    assert err <= TOL[dtype], f"max abs error {err:.2e} > {TOL[dtype]:.0e}"


def skip_unsupported(name, dtype):
    if dtype != torch.float16 and name in cuda_attn.FP16_ONLY:
        pytest.skip(f"{name} is float16 only")
    if name == "fused_tc" and torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("tensor cores need sm_70+")


@cuda
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n", SHAPES_N)
@pytest.mark.parametrize("causal", [False, True])
def test_matches_reference(name, dtype, n, causal):
    skip_unsupported(name, dtype)
    q, k, v = make_qkv(2, 4, n, 64, dtype)
    out = cuda_attn.KERNELS[name](q, k, v, causal)
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == dtype
    assert_close(out, cuda_attn.reference(q, k, v, causal), dtype)


@cuda
@pytest.mark.parametrize("br,bc", [(64, 32), (64, 64), (128, 32), (128, 64)])
@pytest.mark.parametrize("double_buffer", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_tuned_tile_configs(br, bc, double_buffer, causal):
    q, k, v = make_qkv(2, 3, 333, 64, torch.float16)
    try:
        out = cuda_attn.fused_tuned(q, k, v, causal, double_buffer, br, bc)
    except RuntimeError as e:
        if "shared memory" in str(e):
            pytest.skip(str(e))  # e.g. (128, 64) double-buffered exceeds a T4's 64 KB
        raise
    assert_close(out, cuda_attn.reference(q, k, v, causal), torch.float16)


@cuda
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("scale", [10.0, 30.0])
def test_large_scores_do_not_overflow(name, scale):
    dtype = torch.float16 if name in cuda_attn.FP16_ONLY else torch.float32
    skip_unsupported(name, dtype)
    q, k, v = make_qkv(1, 2, 256, 64, dtype)
    q, k = (q.float() * scale).to(dtype), (k.float() * scale).to(dtype)
    out = cuda_attn.KERNELS[name](q, k, v)
    assert torch.isfinite(out).all()
    # Softmax becomes nearly one-hot at this scale, so compare with a looser tolerance.
    assert (out.float() - cuda_attn.reference(q, k, v)).abs().max().item() < 1e-2


@cuda
def test_long_sequence_fused_versions_agree():
    """At N = 8192 the reference itself is expensive; check the fused versions against each other."""
    q, k, v = make_qkv(1, 2, 8192, 64, torch.float16)
    base = cuda_attn.fused(q, k, v).float()
    for name in ("fused_vec", "fused_db", "fused_tc"):
        skip_unsupported(name, torch.float16)
        assert (cuda_attn.KERNELS[name](q, k, v).float() - base).abs().max().item() < 2e-3, name


@cuda
@pytest.mark.parametrize("name", ["naive", "tiled"])
@pytest.mark.parametrize("d", [32, 80, 128])
def test_other_head_dims_for_unfused(name, d):
    q, k, v = make_qkv(1, 2, 200, d, torch.float32)
    assert_close(cuda_attn.KERNELS[name](q, k, v), cuda_attn.reference(q, k, v), torch.float32)


@cuda
@pytest.mark.parametrize("name", ["fused", "fused_vec", "fused_tc"])
def test_fused_rejects_unsupported_head_dim(name):
    q, k, v = make_qkv(1, 1, 64, 128, torch.float16)
    with pytest.raises(RuntimeError, match="head dimension"):
        cuda_attn.KERNELS[name](q, k, v)


@cuda
def test_fp16_only_kernels_reject_fp32():
    q, k, v = make_qkv(1, 1, 64, 64, torch.float32)
    for name in cuda_attn.FP16_ONLY:
        with pytest.raises(RuntimeError, match="float16"):
            cuda_attn.KERNELS[name](q, k, v)


@cuda
def test_input_validation():
    q, k, v = make_qkv(1, 1, 64, 64, torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        cuda_attn.fused(q.transpose(2, 3), k, v)
    with pytest.raises(RuntimeError, match="dtype"):
        cuda_attn.fused(q, k.float(), v)
    with pytest.raises(RuntimeError, match="not compiled"):
        cuda_attn.fused_tuned(q, k, v, br=32, bc=32)
