"""Run attention kernels on the GPGPU-Sim cycle-level simulator and collect stats as CSV.

    python sim/run_sim.py --config sim/configs/A100_slice --jobs 2 \
        --runs "fused:42:128:0" "fused_tc:42:128:0" ... --out results/simulated/perf.csv

Each run is  kernel:BH:N:causal . Every run executes in its own scratch directory (the
simulator writes files to its working directory) with a copy of the GPU config.

Collected per run: correctness vs a double-precision CPU reference (computed by the harness),
total simulated cycles across the run's kernel launches, warp instructions, achieved
occupancy, L2 accesses and misses, shared-memory and global load instruction counts.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

STATS = {  # name in CSV -> regex on the simulator log (last occurrence wins: totals after the final kernel)
    "cycles": r"^gpu_tot_sim_cycle = (\d+)",
    "thread_insn": r"^gpu_tot_sim_insn = (\d+)",
    "warp_insn": r"^gpgpu_n_tot_w_icount = (\d+)",
    "occupancy_pct": r"^gpu_occupancy = ([\d.]+)%",
    "l2_accesses": r"^L2_total_cache_accesses = (\d+)",
    "l2_misses": r"^L2_total_cache_misses = (\d+)",
    "shmem_insn": r"^gpgpu_n_shmem_insn = (\d+)",
    "global_load_insn": r"^gpgpu_n_load_insn\s+= (\d+)",
    "dram_reads": r"^total dram reads = (\d+)",
    "dram_writes": r"^total dram writes = (\d+)",
    "shmem_bank_conflicts": r"^gpgpu_n_shmem_bkconflict = (\d+)",
    "scoreboard_stall_warp_cycles": r"W0_Scoreboard:(\d+)",
    "idle_warp_cycles": r"W0_Idle:(\d+)",
}
KERNEL_REGS = r"Kernel '(\S+)' : regs=(\d+), lmem=(\d+), smem=(\d+)"
RESULT = r"RESULT kernel=(\S+) BH=(\d+) N=(\d+) causal=(\d) max_abs_err=(\S+) status=(\w+)"


def parse(log: str) -> dict:
    out = {}
    for key, rx in STATS.items():
        hits = re.findall(rx, log, flags=re.M)
        out[key] = hits[-1] if hits else ""
    m = re.findall(RESULT, log)
    if m:
        out["max_abs_err"], out["status"] = m[-1][4], m[-1][5]
    else:
        err = re.findall(r"RESULT status=(\S+).*", log)
        out["status"] = err[-1] if err else "NO_RESULT"
        out["max_abs_err"] = ""
    out["kernel_launches"] = len(re.findall(r"^kernel_launch_uid = ", log, flags=re.M))
    return out


def run_one(spec: str, harness: str, config_dir: str, env: dict, keep_logs: str | None) -> dict:
    parts = spec.split(":")
    kernel = ":".join(parts[:-3])  # "fused" or "fused_tuned:br:bc:db"
    bh, n, causal = parts[-3:]
    work = tempfile.mkdtemp(prefix="sim_")
    for f in os.listdir(config_dir):
        shutil.copy(os.path.join(config_dir, f), work)
    t0 = time.time()
    proc = subprocess.run([harness, kernel, bh, n, causal], cwd=work, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    log = proc.stdout
    if keep_logs:
        os.makedirs(keep_logs, exist_ok=True)
        with open(os.path.join(keep_logs, spec.replace(":", "_") + ".log"), "w") as fh:
            fh.write(log)
    row = dict(kernel=kernel, BH=bh, N=n, causal=causal, sim_wall_s=f"{wall:.0f}", **parse(log))
    regs = re.findall(KERNEL_REGS, log)
    row["regs"] = ";".join(sorted({r[1] for r in regs}))
    shutil.rmtree(work, ignore_errors=True)
    print(f"{spec:28} {row['status']:9} cycles={row['cycles']:>10} err={row['max_abs_err']} ({wall:.0f}s)", flush=True)
    return row


def reparse(log_dir: str, out: str):
    """Rebuild a CSV from kept simulator logs (e.g. after adding a statistic)."""
    rows = []
    for f in sorted(os.listdir(log_dir)):
        log = open(os.path.join(log_dir, f)).read()
        m = re.findall(RESULT, log)
        if not m:
            continue
        kernel, bh, n, causal = m[-1][:4]
        rows.append(dict(kernel=kernel, BH=bh, N=n, causal=causal, **parse(log)))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", default=os.path.join(HERE, "harness"))
    p.add_argument("--config", default=os.path.join(HERE, "configs", "A100_slice"))
    p.add_argument("--runs", nargs="+", default=[])
    p.add_argument("--reparse", default=None, help="rebuild --out from the logs in this directory, then exit")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", default="results/simulated/runs.csv")
    p.add_argument("--keep-logs", default=None)
    args = p.parse_args()
    if args.reparse:
        reparse(args.reparse, args.out)
        return

    env = dict(os.environ)  # caller must have sourced GPGPU-Sim's setup_environment
    assert "gpgpu-sim" in env.get("LD_LIBRARY_PATH", ""), "source gpgpu-sim setup_environment first"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["kernel", "BH", "N", "causal", "status", "max_abs_err", "cycles", "warp_insn", "thread_insn",
              "occupancy_pct", "l2_accesses", "l2_misses", "dram_reads", "dram_writes", "shmem_insn",
              "global_load_insn", "shmem_bank_conflicts", "scoreboard_stall_warp_cycles", "idle_warp_cycles",
              "kernel_launches", "regs", "sim_wall_s"]
    new = not os.path.exists(args.out)
    with ThreadPoolExecutor(args.jobs) as pool, open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in pool.map(lambda s: run_one(s, args.harness, args.config, env, args.keep_logs), args.runs):
            w.writerow(row)
            fh.flush()


if __name__ == "__main__":
    main()
