"""CPU-only checks of the online-softmax algorithm the fused kernel implements.

Runs anywhere (no GPU). If these pass and the GPU tests fail, the bug is in the CUDA code,
not in the math.
"""
import numpy as np
import pytest

from cuda_attn.online_softmax_ref import attention_naive, attention_online


@pytest.mark.parametrize("n", [1, 31, 32, 63, 64, 65, 100, 257])
@pytest.mark.parametrize("causal", [False, True])
def test_online_matches_naive(n, causal):
    rng = np.random.default_rng(n)
    q, k, v = (rng.standard_normal((2, 3, n, 64)).astype(np.float32) for _ in range(3))
    np.testing.assert_allclose(attention_online(q, k, v, causal), attention_naive(q, k, v, causal),
                               atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("scale", [10.0, 100.0])
def test_online_is_stable_for_large_scores(scale):
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((1, 1, 200, 64)).astype(np.float32) for _ in range(3))
    out = attention_online(q * scale, k * scale, v)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, attention_naive(q * scale, k * scale, v), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("br,bc", [(16, 16), (64, 32), (128, 64)])
def test_tile_size_does_not_change_result(br, bc):
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal((1, 2, 300, 64)).astype(np.float32) for _ in range(3))
    np.testing.assert_allclose(attention_online(q, k, v, True, br, bc), attention_naive(q, k, v, True),
                               atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("n", [1, 64, 65, 1000, 4096])
@pytest.mark.parametrize("causal", [False, True])
def test_fp16_p_stays_within_test_tolerance(n, causal):
    """v4 rounds P to fp16 for the tensor-core matmul; the error must fit the GPU tests' 2e-3 bound."""
    rng = np.random.default_rng(7)
    q, k, v = (rng.standard_normal((1, 2, n, 64)).astype(np.float16).astype(np.float32) for _ in range(3))
    out = attention_online(q, k, v, causal, br=64, bc=64, p_fp16=True)
    out = out.astype(np.float16).astype(np.float32)  # the kernel also writes O in fp16
    assert np.abs(out - attention_naive(q, k, v, causal)).max() <= 2e-3
