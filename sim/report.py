"""Turn simulator CSVs into the Simulated results page and chart.

    python sim/report.py --correctness results/simulated/correctness.csv \
                         --perf results/simulated/perf.csv --out results/simulated

Every number this writes is labeled **Simulated**: it comes from GPGPU-Sim running the real
kernels (PTX) on the SM80_A100 model, 14-SM slice. Real-GPU numbers from bench/benchmark.py
are labeled **Measured** and live in results/measured/.

Derived columns (all from simulated cycles, no other model):
  time_ms           cycles / 1410 MHz (the config's core clock)
  tflops_slice      4*BH*N^2*d FLOPs (halved if causal) / time
  tflops_a100_eq    tflops_slice * 108/14: the slice keeps per-SM bandwidth and L2, so the full
                    chip runs 108/14 as many blocks in the same time (needs several waves of blocks)
  pct_of_peak       vs the peak of the units the kernel computes on:
                    fp32   (v1-v3.x): 14 SMs x 64 FP32 lanes x 2 x 1.41 GHz = 2.53 TFLOP/s
                    tensor (v4):      312 TFLOP/s x 14/108 = 40.4 TFLOP/s (A100 dense FP16 tensor peak)
  speedup_vs_naive  naive cycles / kernel cycles at the same shape
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

CLOCK_HZ = 1.41e9
SLICE_SMS, A100_SMS = 14, 108
FP32_PEAK_SLICE = SLICE_SMS * 64 * 2 * CLOCK_HZ
TENSOR_PEAK_SLICE = 312e12 * SLICE_SMS / A100_SMS
D = 64

ORDER = ["naive", "tiled", "fused", "fused_vec", "fused_db", "fused_tc"]
LABEL = {"naive": "v1 naive", "tiled": "v2 tiled", "fused": "v3 fused",
         "fused_vec": "v3.1 128-bit loads", "fused_db": "v3.2 double buffer*", "fused_tc": "v4 tensor cores"}


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def derive(rows):
    out = []
    base = {}
    for r in rows:
        if r["status"] == "PASS" and r["kernel"] == "naive":
            base[(r["BH"], r["N"], r["causal"])] = int(r["cycles"])
    for r in rows:
        if r["status"] != "PASS" or not r["cycles"]:
            continue
        bh, n, causal = int(r["BH"]), int(r["N"]), r["causal"] == "1"
        cyc = int(r["cycles"])
        t = cyc / CLOCK_HZ
        flops = 4 * bh * n * n * D * (0.5 if causal else 1)
        tf = flops / t
        nb = base.get((r["BH"], r["N"], r["causal"]))
        out.append(dict(
            label="Simulated", kernel=r["kernel"], version=LABEL.get(r["kernel"], r["kernel"]),
            BH=bh, N=n, causal=int(causal), cycles=cyc, time_ms=f"{t * 1e3:.3f}",
            tflops_slice=f"{tf / 1e12:.3f}", tflops_a100_eq=f"{tf / 1e12 * A100_SMS / SLICE_SMS:.2f}",
            peak_unit="tensor" if r["kernel"] == "fused_tc" else "fp32",
            pct_of_peak=f"{100 * tf / (TENSOR_PEAK_SLICE if r['kernel'] == 'fused_tc' else FP32_PEAK_SLICE):.1f}",
            speedup_vs_naive=f"{nb / cyc:.1f}" if nb else "",
            occupancy_pct=r.get("occupancy_pct", ""), warp_insn=r.get("warp_insn", ""),
            dram_reads=r.get("dram_reads", ""), dram_writes=r.get("dram_writes", ""),
            l2_misses=r.get("l2_misses", ""), shmem_bank_conflicts=r.get("shmem_bank_conflicts", ""),
            max_abs_err=r["max_abs_err"]))
    out.sort(key=lambda x: (x["causal"], x["N"], ORDER.index(x["kernel"]) if x["kernel"] in ORDER else 99))
    return out


def chart(rows, path, n, causal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = {r["kernel"]: r for r in rows if r["N"] == n and r["causal"] == causal and r["speedup_vs_naive"]}
    ks = [k for k in ORDER if k in sel]
    if len(ks) < 2:
        return False
    vals = [float(sel[k]["speedup_vs_naive"]) for k in ks]
    best = ks[vals.index(max(vals))]
    ink, quiet, grey, accent = "#1f1f1f", "#6b6b6b", "#b8b8b8", "#2a6fdb"
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(ks) + 1.6), dpi=160)
    y = list(range(len(ks)))[::-1]
    ax.barh(y, vals, height=0.6, color=[accent if k == best else grey for k in ks])
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f"{v:.1f}x", va="center", fontsize=10, color=ink)
    ax.set_yticks(y, [LABEL[k] for k in ks], fontsize=10, color=ink)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("speedup over v1 naive (higher is faster)", color=quiet)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grey)
    ax.tick_params(axis="x", colors=quiet)
    ax.tick_params(axis="y", length=0)
    title = f"Simulated: {LABEL[best]} is {max(vals):.0f}x faster than naive"
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=ink)
    bh = sel[ks[0]]["BH"]
    fig.text(0.01, 0.01, f"GPGPU-Sim, A100 model (14-SM slice) · BH={bh}, N={n}, d=64, "
             f"{'causal' if causal else 'non-causal'}, FP16"
             + (" · *v3.2 runs its synchronous fallback (no cp.async in the simulator)" if "fused_db" in ks else ""),
             fontsize=7, color=quiet)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--correctness", required=True)
    p.add_argument("--perf", required=True)
    p.add_argument("--out", default="results/simulated")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    corr = read(a.correctness)
    perf = derive(read(a.perf))
    with open(os.path.join(a.out, "perf_derived.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(perf[0]))
        w.writeheader()
        w.writerows(perf)

    by_k = defaultdict(list)
    for r in corr:
        by_k[r["kernel"]].append(r)
    lines = ["# Results: Simulated", "",
             "Every number on this page is **Simulated**: GPGPU-Sim (accel-sim fork, dev @ 4195dd4, with the",
             "fixes in `sim/patches/`) executing the kernels' PTX on its SM80_A100 model, cut to a 14-SM slice",
             "(5 of 40 memory channels, same per-SM bandwidth and L2). Real-GPU numbers will be labeled **Measured**.",
             "", "## Correctness (Simulated)", "",
             "| Version | Runs | Passed | Max abs error | Shapes |", "|---|---|---|---|---|"]
    for k in ORDER:
        rs = by_k.get(k, [])
        if not rs:
            continue
        ok = sum(r["status"] == "PASS" for r in rs)
        err = max(float(r["max_abs_err"]) for r in rs if r["max_abs_err"])
        shapes = ", ".join(sorted({f"N={r['N']}{' causal' if r['causal'] == '1' else ''}" for r in rs}))
        lines.append(f"| {LABEL[k]} | {len(rs)} | {ok} | {err:.1e} | BH=2; {shapes} |")
    lines += ["", "Pass = max abs error ≤ 2e-3 vs a float64 CPU reference on the same FP16 inputs.", ""]

    lines += ["## Performance (Simulated)", "",
              "| Version | BH | N | Causal | Cycles | Time (ms, slice) | TFLOP/s (slice) | TFLOP/s, A100-equivalent | % of peak | Speedup vs naive | Occupancy |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in perf:
        lines.append(f"| {r['version']} | {r['BH']} | {r['N']} | {'yes' if r['causal'] else 'no'} | {r['cycles']:,} | "
                     f"{r['time_ms']} | {r['tflops_slice']} | {r['tflops_a100_eq']} | {r['pct_of_peak']}% {r['peak_unit']} | "
                     f"{r['speedup_vs_naive'] + 'x' if r['speedup_vs_naive'] else 'naive not run'} | "
                     f"{float(r['occupancy_pct'] or 0):.1f}% |")
    lines += ["", "% of peak compares against the units each kernel computes on: FP32 cores for v1 to v3.1, tensor",
              "cores for v4. A100-equivalent = slice TFLOP/s x 108/14 (the slice keeps per-SM bandwidth and L2).",
              "Naive was simulated only at N=128: at longer sequences it would take many hours of simulation.", "",
              "*v3.2 runs its synchronous fallback: the simulator does not implement `cp.async`, so its",
              "double buffering cannot help here. It needs a Measured run.", ""]

    shapes = sorted({(r["N"], r["causal"]) for r in perf})
    for n, c in shapes:
        name = f"speedup_N{n}{'_causal' if c else ''}.png"
        if chart(perf, os.path.join(a.out, name), n, c):
            lines.append(f"![Simulated speedup, N={n}{' causal' if c else ''}]({name})")
            lines.append("")

    lines += ["## What Simulated can and cannot say", "",
              "- It runs the exact kernel code, so correctness results and the *ordering* of versions are solid.",
              "- GPGPU-Sim executes PTX, not the SASS that ptxas would produce (no instruction scheduling or",
              "  register-level optimization), so absolute times are estimates. Ratios between versions are more",
              "  trustworthy than any single TFLOP/s value.",
              "- FlashAttention-2 and PyTorch SDPA cannot run in the simulator, so **% of FlashAttention-2 is a",
              "  Measured-only result**.", ""]
    with open(os.path.join(a.out, "README.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
