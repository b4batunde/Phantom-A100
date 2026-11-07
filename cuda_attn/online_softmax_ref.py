"""Pure-NumPy model of the v3 fused kernel.

Mirrors attn_fused.cu step by step (tiles of BC keys, running max m, running sum l,
log2-domain scores) so the algorithm can be checked on a laptop without a GPU, and so a
kernel bug can be told apart from an algorithm bug.
"""
from __future__ import annotations

import numpy as np

LOG2E = 1.4426950408889634


def attention_naive(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    """Textbook softmax(QK^T / sqrt(d)) V on (..., N, d) arrays, in float64."""
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    n, d = q.shape[-2], q.shape[-1]
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    if causal:
        s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def attention_online(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False,
                     br: int = 64, bc: int = 32, p_fp16: bool = False) -> np.ndarray:
    """Tile-by-tile online softmax, exactly as the fused kernels compute it (float32).

    p_fp16=True models v4 (tensor cores): P is rounded to float16 before the P @ V matmul,
    while the row sum l is accumulated from the unrounded float32 values.
    """
    q, k, v = (x.astype(np.float32) for x in (q, k, v))
    *batch, n, d = q.shape
    out = np.empty_like(q)
    scale_log2 = np.float32(LOG2E / np.sqrt(d))

    for i0 in range(0, n, br):                      # one thread block
        rows = slice(i0, min(i0 + br, n))
        qi = q[..., rows, :] * scale_log2           # pre-scaled, log2 domain
        r = qi.shape[-2]
        row_idx = np.arange(i0, i0 + r)
        m = np.full((*batch, r), -np.inf, np.float32)
        l = np.zeros((*batch, r), np.float32)
        o = np.zeros((*batch, r, d), np.float32)
        kv_end = min(n, i0 + br) if causal else n
        for j0 in range(0, kv_end, bc):             # stream K/V tiles
            cols = slice(j0, min(j0 + bc, n))
            s = qi @ np.swapaxes(k[..., cols, :], -1, -2)
            if causal:
                col_idx = np.arange(j0, cols.stop)
                s = np.where(col_idx[None, :] > row_idx[:, None], -np.inf, s)
            m_new = np.maximum(m, s.max(axis=-1))
            with np.errstate(invalid="ignore"):
                alpha = np.where(np.isneginf(m_new), 0.0, np.exp2(m - m_new)).astype(np.float32)
                p = np.where(np.isneginf(m_new)[..., None], 0.0, np.exp2(s - m_new[..., None])).astype(np.float32)
            l = l * alpha + p.sum(axis=-1)
            p_mm = p.astype(np.float16).astype(np.float32) if p_fp16 else p
            o = o * alpha[..., None] + p_mm @ v[..., cols, :]
            m = m_new
        out[..., rows, :] = o / l[..., None]
    return out
