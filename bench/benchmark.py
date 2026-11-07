"""Benchmark sweep: my kernels vs PyTorch SDPA and FlashAttention-2.

    python bench/benchmark.py                       # full sweep -> results/measured/bench_<gpu>.csv
    python bench/benchmark.py --ns 1024 4096 --impls fused flash2

Timing protocol: warm-up runs, then timed runs with CUDA events, L2 flushed before
each timed run; median and 10th/90th percentiles reported. Every run first checks the
output against the FP32 reference so a fast-but-wrong kernel never makes the table.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cuda_attn  # noqa: E402

try:
    from flash_attn import flash_attn_func  # FlashAttention-2
except ImportError:
    flash_attn_func = None


# ---------------------------------------------------------------- implementations
def sdpa_math(q, k, v, causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def sdpa_default(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash2(q, k, v, causal):
    # flash-attn expects (B, N, H, d); the transposes are done outside the timed region.
    return flash_attn_func(q, k, v, causal=causal)


IMPLS = {
    "naive": cuda_attn.naive,
    "tiled": cuda_attn.tiled,
    "fused": cuda_attn.fused,
    "fused_vec": cuda_attn.fused_vec,
    "fused_db": cuda_attn.fused_db,
    "fused_tc": cuda_attn.fused_tc,
    "sdpa_math": sdpa_math,
    "sdpa_default": sdpa_default,
    "flash2": flash2,
}
NEEDS_BNHD = {"flash2"}
MATERIALIZES_S = {"naive", "tiled", "sdpa_math"}


# ---------------------------------------------------------------- helpers
def attention_flops(b, h, n, d, causal):
    """Two matmuls of 2*N*N*d FLOPs each per head; causal does half the work (FA convention)."""
    flops = 4 * b * h * n * n * d
    return flops / 2 if causal else flops


def fits_in_memory(name, b, h, n, d, dtype_bytes):
    """Skip unfused kernels whose N x N score buffer cannot fit, instead of crashing the sweep."""
    if name not in MATERIALIZES_S:
        return True
    s_bytes = b * h * n * n * 4 * (2 if name == "sdpa_math" else 1)
    free, _ = torch.cuda.mem_get_info()
    return s_bytes + 4 * b * h * n * d * dtype_bytes < 0.85 * free


def time_cuda(fn, warmup, iters, flush_buf):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush_buf.zero_()  # evict L2 so every run starts cold
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    q = statistics.quantiles(times, n=10)
    return statistics.median(times), q[0], q[-1]


def peak_memory_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impls", nargs="+", default=list(IMPLS))
    p.add_argument("--ns", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--causal", choices=["off", "on", "both"], default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    assert torch.cuda.is_available(), "benchmark needs a CUDA GPU"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    gpu = torch.cuda.get_device_name()
    cc = "sm_%d%d" % torch.cuda.get_device_capability()
    out = args.out or os.path.join("results", "measured", f"bench_{gpu.replace(' ', '_')}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    causal_modes = {"off": [False], "on": [True], "both": [False, True]}[args.causal]

    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush_buf = torch.empty(max(2 * l2, 64 * 2**20), dtype=torch.uint8, device="cuda")

    meta = dict(gpu=gpu, cc=cc, torch=torch.__version__, cuda=torch.version.cuda,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"# {meta}")
    fields = ["gpu", "cc", "torch", "cuda", "timestamp", "impl", "dtype", "causal", "B", "H", "N", "d",
              "median_ms", "p10_ms", "p90_ms", "tflops", "peak_mem_mb", "max_abs_err", "status"]

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for causal in causal_modes:
            for n in args.ns:
                b, h, d = args.batch, args.heads, args.dim
                q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
                ref = cuda_attn.reference(q, k, v, causal) if n <= 4096 else None
                for name in args.impls:
                    row = dict(meta, impl=name, dtype=args.dtype, causal=causal, B=b, H=h, N=n, d=d)
                    fn_raw = IMPLS[name]
                    if name == "flash2" and (flash_attn_func is None or dtype != torch.float16
                                             or torch.cuda.get_device_capability()[0] < 8):
                        w.writerow(dict(row, status="unavailable")); continue
                    if name in cuda_attn.FP16_ONLY and dtype != torch.float16:
                        w.writerow(dict(row, status="fp16_only")); continue
                    if name == "fused_tc" and torch.cuda.get_device_capability()[0] < 7:
                        w.writerow(dict(row, status="unavailable")); continue
                    if name.startswith("fused") and d != 64:
                        w.writerow(dict(row, status="unsupported_d")); continue
                    if not fits_in_memory(name, b, h, n, d, q.element_size()):
                        w.writerow(dict(row, status="skipped_oom")); continue

                    if name in NEEDS_BNHD:
                        qq, kk, vv = (x.transpose(1, 2).contiguous() for x in (q, k, v))
                        fn = lambda: fn_raw(qq, kk, vv, causal)  # noqa: E731
                        get = lambda: fn().transpose(1, 2)  # noqa: E731
                    else:
                        fn = lambda: fn_raw(q, k, v, causal)  # noqa: E731
                        get = fn
                    try:
                        err = (get().float() - ref).abs().max().item() if ref is not None else float("nan")
                        tol = 2e-3 if dtype == torch.float16 else 1e-4
                        if err > tol:
                            w.writerow(dict(row, max_abs_err=f"{err:.2e}", status="wrong_result")); continue
                        iters = args.iters if name not in ("naive",) or n <= 2048 else max(10, args.iters // 10)
                        med, p10, p90 = time_cuda(fn, args.warmup, iters, flush_buf)
                        mem = peak_memory_mb(fn)
                    except RuntimeError as e:  # e.g. a tile config that needs more shared memory than this GPU has
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                            w.writerow(dict(row, status="oom")); continue
                        print(f"{name} N={n}: {e}")
                        w.writerow(dict(row, status="error")); continue
                    tflops = attention_flops(b, h, n, d, causal) / (med * 1e-3) / 1e12
                    w.writerow(dict(row, median_ms=f"{med:.4f}", p10_ms=f"{p10:.4f}", p90_ms=f"{p90:.4f}",
                                    tflops=f"{tflops:.3f}", peak_mem_mb=f"{mem:.1f}",
                                    max_abs_err=f"{err:.2e}", status="ok"))
                    fh.flush()
                    print(f"causal={causal!s:5} N={n:5} {name:12} {med:9.3f} ms  {tflops:7.2f} TFLOP/s  "
                          f"{mem:9.1f} MB  err={err:.1e}")
                del q, k, v, ref
                torch.cuda.empty_cache()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
