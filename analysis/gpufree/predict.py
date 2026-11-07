"""Predict full-A100 runtimes for every kernel from simulator runs + the analytical model.

    cd analysis && python -m gpufree.predict results/simulated/perf.csv

Method
------
1. The simulator ran each kernel on a 1/8 "slice" of an A100 (14 SMs, 5 of 40 memory channels:
   same per-SM bandwidth and L2) with exactly one full wave of thread blocks. Blocks are
   independent, so one wave on the slice takes as long as one wave on the full chip.
2. Fused kernels (v3-v4): each block streams N keys, so cycles per wave are linear in N.
   Two simulated sequence lengths give  cycles(N) = a + b*N.  Full-chip time for any shape is
       waves * cycles(N) / clock,   waves = ceil(blocks / (108 SMs * blocks_per_SM)).
3. Unfused kernels (v1-v2): their N x N buffer fits in L2 at the simulated size but not at real
   sizes, so the simulated cycles measure everything except DRAM; the model's DRAM time is added.
4. Every prediction is a RANGE:
       low throughput  = the simulator's (PTX executes in order, without ptxas scheduling: pessimistic)
       high throughput = the analytical ideal bound (perfect overlap, no stalls: optimistic)
5. Efficiency = ideal analytical bound / simulated time, i.e. how close each kernel gets to its
   own roofline; it explains WHY a kernel is slow (latency, bank conflicts, occupancy).

Everything here is a prediction. The real-GPU benchmark (bench/benchmark.py) supersedes it.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace

from .model import A100_80GB, COSTS, bound, occupancy, useful_flops
from .reproduce_fa import FA2_FWD_D64

SLICE = replace(A100_80GB, name="A100 slice (14 SMs)", sms=14, hbm_gbps=A100_80GB.hbm_gbps * 5 / 40)
FA2_NS = [512, 1024, 2048, 4096, 8192, 16384]  # FA2 Fig. 5 shapes: 16k tokens, 32 heads, d=64
FUSED = ["fused", "fused_vec", "fused_db", "fused_tc"]
UNFUSED = ["naive", "tiled"]


def fa2_shape(n):
    return (16384 // n) * 32, n  # BH, N


def load_sim(path):
    runs = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["status"] != "PASS" or not r["cycles"]:
            continue
        runs[r["kernel"]].append(dict(BH=int(r["BH"]), N=int(r["N"]), cycles=int(r["cycles"]),
                                      occupancy=float(r["occupancy_pct"] or 0), warp_insn=int(r["warp_insn"] or 0),
                                      l2_misses=int(r["l2_misses"] or 0), l2_accesses=int(r["l2_accesses"] or 0)))
    return runs


def fit_linear(points):
    """cycles = a + b * N through the simulated points (exact for two points, least squares otherwise)."""
    xs, ys = [p["N"] for p in points], [p["cycles"] for p in points]
    if len(points) == 1:
        return 0.0, ys[0] / xs[0]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def predict(path, out_dir="analysis"):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_sim(path)
    gpu, hz = A100_80GB, A100_80GB.clock_ghz * 1e9
    eff_rows, pred_rows = [], []

    for k, pts in sorted(runs.items()):
        c = COSTS.get(k)
        if c is None:
            continue
        occ = occupancy(gpu, c.launch)
        for p in pts:
            ideal = bound(SLICE, k, p["BH"], p["N"])
            sim_ms = p["cycles"] / hz * 1e3
            eff_rows.append(dict(kernel=k, label=c.label, BH=p["BH"], N=p["N"], sim_cycles=p["cycles"],
                                 sim_ms=f"{sim_ms:.3f}", ideal_ms=f"{ideal['time_ms']:.3f}",
                                 ideal_limiter=ideal["limiter"], efficiency=f"{ideal['time_ms'] / sim_ms:.3f}",
                                 sim_tflops_slice=f"{useful_flops(p['BH'], p['N']) / (sim_ms / 1e3) / 1e12:.2f}",
                                 sim_occupancy_pct=p["occupancy"], model_occupancy_pct=f"{100 * occ['occupancy']:.1f}",
                                 occupancy_limited_by=occ["limited_by"]))

        if k in FUSED:
            a, b = fit_linear(pts)
            per_wave = 108 * occ["blocks_per_sm"]
            for n in FA2_NS:
                bh, n_ = fa2_shape(n)
                blocks = bh * math.ceil(n_ / c.launch.rows_per_block)
                waves = math.ceil(blocks / per_wave)
                t_sim = waves * (a + b * n_) / hz                     # pessimistic: PTX-level simulation
                t_ideal = bound(gpu, k, bh, n_)["time_ms"] / 1e3       # optimistic: analytical roofline
                f = useful_flops(bh, n_)
                pred_rows.append(dict(kernel=k, label=c.label, N=n_, BH=bh, low_ms=t_ideal * 1e3, high_ms=t_sim * 1e3,
                                      tflops_low=f / t_sim / 1e12, tflops_high=f / t_ideal / 1e12, waves=waves,
                                      basis=f"low: sim ({len(pts)} points, cycles/wave = {a:.0f} + {b:.1f} N); high: ideal bound"))
        else:
            # SM-cycles per score measured at the simulated size (S resident in L2) ...
            p = max(pts, key=lambda q: q["N"])
            sm_cycles_per_score = p["cycles"] * SLICE.sms / (p["BH"] * p["N"] ** 2)
            for n in FA2_NS:
                bh, n_ = fa2_shape(n)
                t_sim = sm_cycles_per_score * bh * n_ ** 2 / gpu.sms / hz
                t_dram = bound(gpu, k, bh, n_)["t_dram_ms"] / 1e3   # ... plus the N x N traffic L2 can't hold
                t_pess = t_sim + t_dram                                 # pessimistic: sim + DRAM, no overlap
                t_ideal = bound(gpu, k, bh, n_)["time_ms"] / 1e3       # optimistic: analytical roofline
                f = useful_flops(bh, n_)
                pred_rows.append(dict(kernel=k, label=c.label, N=n_, BH=bh, low_ms=t_ideal * 1e3, high_ms=t_pess * 1e3,
                                      tflops_low=f / t_pess / 1e12, tflops_high=f / t_ideal / 1e12, waves="",
                                      basis=f"low: sim at N={p['N']} ({sm_cycles_per_score:.1f} SM-cycles/score) + DRAM model; high: ideal bound"))

    fa2 = dict(zip(FA2_NS, FA2_FWD_D64[False]))
    for r in pred_rows:
        r["fa2_tflops"] = fa2[r["N"]]
        r["pct_of_fa2_low"] = 100 * r["tflops_low"] / fa2[r["N"]]
        r["pct_of_fa2_high"] = 100 * r["tflops_high"] / fa2[r["N"]]

    with open(os.path.join(out_dir, "sim_efficiency.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eff_rows[0]))
        w.writeheader(); w.writerows(eff_rows)
    with open(os.path.join(out_dir, "predictions_a100.csv"), "w", newline="") as fh:
        fields = ["kernel", "label", "N", "BH", "low_ms", "high_ms", "tflops_low", "tflops_high",
                  "fa2_tflops", "pct_of_fa2_low", "pct_of_fa2_high", "waves", "basis"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in pred_rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    return eff_rows, pred_rows


def summarize(eff_rows, pred_rows, out_dir="analysis"):
    lines = ["# GPU-free predictions: A100 80GB, forward, d=64, non-causal", "",
             "## How close each kernel gets to its own ideal bound (simulated)", "",
             "| Kernel | N | Sim TFLOP/s (slice) | Ideal bound set by | Efficiency vs ideal | Occupancy (model) |",
             "|---|---|---|---|---|---|"]
    for r in eff_rows:
        lines.append(f"| {r['label']} | {r['N']} | {r['sim_tflops_slice']} | {r['ideal_limiter']} | "
                     f"{100 * float(r['efficiency']):.0f}% | {r['model_occupancy_pct']}% ({r['occupancy_limited_by']}) |")
    lines += ["", "## Predicted throughput range vs FlashAttention-2's published numbers (FA2 Fig. 5 shapes)", "",
              "| Kernel | N | Predicted TFLOP/s (simulated to ideal) | FA2 published | % of FA2 |", "|---|---|---|---|---|"]
    for r in pred_rows:
        tf = f"{r['tflops_low']:.1f}" if abs(r["tflops_low"] - r["tflops_high"]) < 0.05 else f"{r['tflops_low']:.1f} to {r['tflops_high']:.1f}"
        pct = f"{r['pct_of_fa2_low']:.1f}%" if abs(r["pct_of_fa2_low"] - r["pct_of_fa2_high"]) < 0.05 else f"{r['pct_of_fa2_low']:.1f} to {r['pct_of_fa2_high']:.1f}%"
        lines.append(f"| {r['label']} | {r['N']} | {tf} | {r['fa2_tflops']} | {pct} |")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "predictions.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    e, p = predict(sys.argv[1] if len(sys.argv) > 1 else "results/simulated/perf.csv")
    summarize(e, p)
