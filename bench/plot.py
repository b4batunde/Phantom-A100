"""Turn a benchmark CSV into plots and the headline numbers.

    python bench/plot.py results/measured/bench_NVIDIA_A100-SXM4-40GB.csv

Writes, next to the CSV:
  *_throughput.png   TFLOP/s vs sequence length, causal off and on
  *_memory.png       peak extra memory vs sequence length
  *_summary.md       the headline sentence and a per-version table, filled from the data
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["naive", "tiled", "fused", "fused_vec", "fused_db", "fused_tc", "sdpa_math", "sdpa_default", "flash2"]
MINE = ["naive", "tiled", "fused", "fused_vec", "fused_db", "fused_tc"]
LABEL = {"naive": "v1 naive", "tiled": "v2 tiled", "fused": "v3 fused",
         "fused_vec": "v3.1 + 128-bit loads", "fused_db": "v3.2 + double buffer", "fused_tc": "v4 tensor cores",
         "sdpa_math": "SDPA math", "sdpa_default": "SDPA default", "flash2": "FlashAttention-2"}
# Grey for references, one blue family for my three versions, so the progression reads as one hue.
STYLE = {"naive": dict(color="#c6dbef", marker="o"), "tiled": dict(color="#9ecae1", marker="o"),
         "fused": dict(color="#6baed6", marker="o"), "fused_vec": dict(color="#4292c6", marker="o"),
         "fused_db": dict(color="#2171b5", marker="o"),
         "fused_tc": dict(color="#08306b", marker="o", linewidth=2.5),
         "sdpa_math": dict(color="#bdbdbd", marker="s", linestyle="--"),
         "sdpa_default": dict(color="#737373", marker="s", linestyle="--"),
         "flash2": dict(color="#252525", marker="^", linestyle="--")}


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok"]
    data = defaultdict(dict)  # (impl, causal) -> {N: row}
    for r in rows:
        data[(r["impl"], r["causal"] == "True")][int(r["N"])] = r
    return rows, data


def plot_metric(data, key, ylabel, title, path, logy=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, causal in zip(axes, (False, True)):
        for impl in ORDER:
            pts = data.get((impl, causal))
            if not pts:
                continue
            ns = sorted(pts)
            ax.plot(ns, [float(pts[n][key]) for n in ns], label=LABEL[impl], **STYLE[impl])
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("sequence length N")
        ax.set_title("causal" if causal else "non-causal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Measured: " + title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary(rows, data):
    gpu = rows[0]["gpu"]
    lines = [f"# Results: Measured on {gpu}", "", "Every number on this page is **Measured** on a real GPU by bench/benchmark.py.", ""]
    fa2, naive = data.get(("flash2", False), {}), data.get(("naive", False), {})
    mine = [i for i in MINE[2:] if data.get((i, False))]
    if mine:
        n_top = max(max(data[(i, False)]) for i in mine)
        best = max((i for i in mine if n_top in data[(i, False)]),
                   key=lambda i: float(data[(i, False)][n_top]["tflops"]))
        fused = data[(best, False)]
        lines.append(f"- Fastest of my kernels at N = {n_top}: **{LABEL[best]}** "
                     f"({float(fused[n_top]['tflops']):.1f} TFLOP/s).")
        if fa2 and set(fused) & set(fa2):
            n = max(set(fused) & set(fa2))
            pct = 100 * float(fused[n]["tflops"]) / float(fa2[n]["tflops"])
            lines.append(f"- {LABEL[best]} reaches **{pct:.0f}% of FlashAttention-2** throughput at N = {n}.")
        if naive and set(fused) & set(naive):
            n = max(set(fused) & set(naive))
            sp = float(naive[n]["median_ms"]) / float(fused[n]["median_ms"])
            mem = float(naive[n]["peak_mem_mb"]) / max(float(fused[n]["peak_mem_mb"]), 1e-9)
            lines.append(f"- {LABEL[best]} is **{sp:.1f}x faster than naive** at N = {n} "
                         f"and uses **{mem:.0f}x less extra memory**.")
    skipped = sorted({int(r["N"]) for r in csv.DictReader(open(sys.argv[1]))
                      if r["impl"] == "naive" and r["status"].startswith("skipped")})
    if skipped:
        lines.append(f"- naive could not run at N = {skipped}: its N x N score matrix does not fit in memory.")
    lines += ["", "| version | N | causal | median ms | TFLOP/s | peak MB |", "|---|---|---|---|---|---|"]
    for impl in ORDER:
        for causal in (False, True):
            for n, r in sorted(data.get((impl, causal), {}).items()):
                lines.append(f"| {LABEL[impl]} | {n} | {causal} | {r['median_ms']} | {r['tflops']} | {r['peak_mem_mb']} |")
    return "\n".join(lines) + "\n"


def main():
    path = sys.argv[1]
    stem = os.path.splitext(path)[0]
    rows, data = load(path)
    if not rows:
        sys.exit("no successful rows in " + path)
    gpu = rows[0]["gpu"]
    plot_metric(data, "tflops", "TFLOP/s", f"Attention forward throughput, {gpu}", stem + "_throughput.png")
    plot_metric(data, "peak_mem_mb", "peak extra memory (MB)", f"Attention memory, {gpu}",
                stem + "_memory.png", logy=True)
    text = summary(rows, data)
    open(stem + "_summary.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
