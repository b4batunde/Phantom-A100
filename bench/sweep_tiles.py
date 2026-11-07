"""Tile-size sweep for the tuned fused kernel ("tile sizes are a sweep, not a guess").

    python bench/sweep_tiles.py                 # N = 1024, 4096, 8192 -> results/measured/tiles_<gpu>.csv

Times every compiled (BR, BC) x {single buffer, double buffer} config and prints the winner
per sequence length. Configs that need more shared memory than the GPU allows are recorded
as skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cuda_attn  # noqa: E402
from benchmark import attention_flops, time_cuda  # noqa: E402

CONFIGS = [(64, 32), (64, 64), (128, 32), (128, 64)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", nargs="+", type=int, default=[1024, 4096, 8192])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    gpu = torch.cuda.get_device_name()
    out = os.path.join("results", "measured", f"tiles_{gpu.replace(' ', '_')}{'_causal' if args.causal else ''}.csv")
    os.makedirs(os.path.join("results", "measured"), exist_ok=True)
    flush = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda")

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gpu", "N", "br", "bc", "double_buffer", "median_ms", "tflops", "status"])
        for n in args.ns:
            q, k, v = (torch.randn(args.batch, args.heads, n, 64, device="cuda", dtype=torch.float16)
                       for _ in range(3))
            ref = cuda_attn.fused(q, k, v, args.causal).float()
            best = None
            for br, bc in CONFIGS:
                for db in (False, True):
                    fn = lambda: cuda_attn.fused_tuned(q, k, v, args.causal, db, br, bc)  # noqa: E731
                    try:
                        if (fn().float() - ref).abs().max().item() > 2e-3:
                            w.writerow([gpu, n, br, bc, db, "", "", "wrong_result"]); continue
                        med, _, _ = time_cuda(fn, 5, args.iters, flush)
                    except RuntimeError as e:
                        w.writerow([gpu, n, br, bc, db, "", "", "skipped: " + str(e).splitlines()[0][:80]])
                        continue
                    tf = attention_flops(args.batch, args.heads, n, 64, args.causal) / (med * 1e-3) / 1e12
                    w.writerow([gpu, n, br, bc, db, f"{med:.4f}", f"{tf:.3f}", "ok"])
                    print(f"N={n:5} BR={br:3} BC={bc:3} double_buffer={db!s:5} {med:8.3f} ms {tf:6.2f} TFLOP/s")
                    if best is None or med < best[0]:
                        best = (med, br, bc, db)
            if best:
                print(f"--> best at N={n}: BR={best[1]} BC={best[2]} double_buffer={best[3]} ({best[0]:.3f} ms)\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
