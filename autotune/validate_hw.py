"""Hardware validation: do the simulator's decisions hold on a real GPU? (Measured)

    python -m autotune.validate_hw                 # on a rented A100 (or any sm_80+ GPU)
    python -m autotune.validate_hw --bh-scale 1    # literal simulated shape instead of the scaled one

What it does:
  1. builds sim/harness natively for the local GPU (same kernels, same dispatch as in simulation)
  2. times every design point in the design space with CUDA events (HARNESS_ITERS launches,
     median), including the double-buffered points the simulator could not judge
  3. compares Measured times with Simulated cycles:
       - Spearman rho and Kendall tau over the sim-verifiable points
       - top-1 agreement, top-3 overlap, and regret (measured time of the simulator's pick /
         measured best)
       - decision accuracy: for every step the guided search judged in simulation, did the sign
         of the speedup hold on hardware?
       - the flagged predictions: did register-capped (spilling) points lose on hardware as the
         trust check assumed? did double buffering help?
Writes results/autotune/measured/{validation.md, times.csv}. Every number there is Measured.

Workload scaling: the simulation ran a 14-SM slice of an A100. With --bh-scale auto (default) the
batch x heads is multiplied by SMs/14 so each SM sees the same number of blocks as in simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess

from .backends import REPO, ReplayBackend
from .space import Workload, all_points

OUT = os.path.join(REPO, "results", "autotune", "measured")
TIME_RE = re.compile(r"TIME kernel=\S+ BH=\d+ N=\d+ causal=\d median_ms=([\d.]+) min_ms=([\d.]+)")
RES_RE = re.compile(r"RESULT .* status=(\w+)")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = a[i] - a[j], b[i] - b[j]
            s += (x * y > 0) - (x * y < 0)
    return s / (n * (n - 1) / 2) if n > 1 else float("nan")


def gpu_info():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,multiprocessor_count,driver_version,clocks.max.sm",
                            "--format=csv,noheader"], stdout=subprocess.PIPE, text=True).stdout.strip()
    except FileNotFoundError:
        q = ""
    return q


def sm_count():
    out = subprocess.run(["python", "-c", "import torch;p=torch.cuda.get_device_properties(0);"
                          "print(p.multi_processor_count, p.major, p.minor)"], stdout=subprocess.PIPE, text=True)
    if out.returncode == 0 and out.stdout.strip():
        sms, ma, mi = out.stdout.split()
        return int(sms), f"{ma}{mi}"
    return 108, "80"


def build(arch: str) -> str:
    exe = os.path.join(REPO, "sim", f"harness_sm{arch}")
    cmd = ["nvcc", "-O3", "-std=c++17", f"-arch=sm_{arch}", "-o", exe, os.path.join(REPO, "sim", "harness.cu")]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return exe


def time_point(exe, p, wl, iters):
    env = dict(os.environ, HARNESS_ITERS=str(iters))
    proc = subprocess.run([exe, p.harness_spec(), str(wl.BH), str(wl.N), str(wl.causal)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    st = RES_RE.findall(proc.stdout)
    t = TIME_RE.findall(proc.stdout)
    return (st[-1] if st else "ERROR"), (float(t[-1][0]) if t else None), proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--BH", type=int, default=42, help="simulated batch x heads")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--bh-scale", default="auto", help="'auto' = SMs/14, or a number")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    sim_wl = Workload(a.BH, a.N, a.causal)
    sms, arch = sm_count()
    scale = sms / 14 if a.bh_scale == "auto" else float(a.bh_scale)
    hw_wl = Workload(max(1, round(a.BH * scale)), a.N, a.causal)
    exe = build(arch)
    be = ReplayBackend()
    exp_path = os.path.join(REPO, "results", "autotune", "experiments.json")
    exp = json.load(open(exp_path)) if os.path.exists(exp_path) else None

    rows = []
    for p in all_points(include_blind=True):
        status, ms, log = time_point(exe, p, hw_wl, a.iters)
        sim = be.cache.get(p, sim_wl)
        rows.append(dict(point=p.key(), label=p.label(), status=status, measured_ms=ms,
                         sim_cycles=sim["cycles"] if sim else None,
                         hw_spill=sim.get("hw_spill_bytes", 0) if sim else None,
                         sim_verifiable=p.sim_verifiable()))
        print(f"{p.key():32} {status:5} {ms if ms is not None else '-':>10} ms   sim={rows[-1]['sim_cycles']}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "times.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    both = [r for r in rows if r["measured_ms"] is not None and r["sim_cycles"] is not None and r["status"] == "PASS"]
    s_cyc = [r["sim_cycles"] for r in both]
    m_ms = [r["measured_ms"] for r in both]
    rho, tau = spearman(s_cyc, m_ms), kendall(s_cyc, m_ms)
    by_sim = sorted(both, key=lambda r: r["sim_cycles"])
    by_hw = sorted(both, key=lambda r: r["measured_ms"])
    trusted_sim = [r for r in by_sim if not r["hw_spill"]]
    top1 = trusted_sim[0]["point"] == by_hw[0]["point"] if trusted_sim else False
    top3 = len({r["point"] for r in by_sim[:3]} & {r["point"] for r in by_hw[:3]})
    regret = trusted_sim[0]["measured_ms"] / by_hw[0]["measured_ms"] if trusted_sim else float("nan")
    ms_of = {r["point"]: r["measured_ms"] for r in rows if r["measured_ms"] is not None}

    decisions = []
    if exp:
        for o in exp["rediscovery"]["outcomes"]:
            if o["parent"] in ms_of and o["child"] in ms_of:
                hw_speed = ms_of[o["parent"]] / ms_of[o["child"]]
                decisions.append(dict(step=o["step"], transform=o["transform"], parent=o["parent"],
                                      child=o["child"], sim_speedup=o["speedup"], hw_speedup=round(hw_speed, 3),
                                      agree=(o["speedup"] > 1) == (hw_speed > 1), trusted=o["trusted"]))

    L = ["# Hardware validation (Measured)", "",
         f"GPU: {gpu_info() or 'unknown'}. Measured workload: BH={hw_wl.BH}, N={hw_wl.N}, causal={hw_wl.causal}, "
         f"d=64, FP16 (simulated workload BH={sim_wl.BH} on a 14-SM slice, scaled by {scale:.2f}). "
         f"Median of {a.iters} launches, CUDA events. Simulated columns come from `results/autotune/sim_cache.json`.",
         "", "## Ranking agreement (sim-verifiable points)", "",
         f"- Points compared: {len(both)}",
         f"- Spearman rho (simulated cycles vs measured time): **{rho:.3f}**",
         f"- Kendall tau: {tau:.3f}",
         f"- Simulator's best trusted point is the measured best: {'yes' if top1 else 'no'}",
         f"- Top-3 overlap: {top3} of 3",
         f"- Regret of trusting the simulator's pick: {regret:.2f}x the measured best time", "",
         "| Design point | Simulated cycles | Sim rank | Measured ms | HW rank | Spills (sm_80) |",
         "|---|---|---|---|---|---|"]
    sim_rank = {r["point"]: i + 1 for i, r in enumerate(by_sim)}
    hw_rank = {r["point"]: i + 1 for i, r in enumerate(by_hw)}
    for r in by_hw:
        L.append(f"| {r['label']} | {r['sim_cycles']:,} | {sim_rank[r['point']]} | {r['measured_ms']:.4f} | "
                 f"{hw_rank[r['point']]} | {r['hw_spill']} B |")
    L += ["", "## Points the simulator could not judge", "", "| Design point | Measured ms | Status |", "|---|---|---|"]
    for r in rows:
        if not r["sim_verifiable"]:
            L.append(f"| {r['label']} | {r['measured_ms'] if r['measured_ms'] is not None else '-'} | {r['status']} |")
    if decisions:
        acc = sum(d["agree"] for d in decisions) / len(decisions)
        L += ["", "## Decision accuracy of the guided search", "",
              f"Of {len(decisions)} transformations the guided search judged in simulation, the direction of the "
              f"speedup held on hardware for **{100 * acc:.0f}%**.", "",
              "| Step | Transformation | Parent -> child | Simulated speedup | Measured speedup | Same direction | Trusted in sim |",
              "|---|---|---|---|---|---|---|"]
        for d in decisions:
            L.append(f"| {d['step']} | `{d['transform']}` | `{d['parent']}` -> `{d['child']}` | {d['sim_speedup']:.2f}x | "
                     f"{d['hw_speedup']:.2f}x | {'yes' if d['agree'] else '**no**'} | {'yes' if d['trusted'] else 'no'} |")
    open(os.path.join(OUT, "validation.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
