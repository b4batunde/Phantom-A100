"""Hardware counters from Nsight Compute, and the explicit simulator -> hardware metric mapping.

GPGPU-Sim counters and Nsight Compute metrics are not the same measurements under different names.
MAPPING states, for every simulator counter the optimizer reasons about, which hardware metric(s)
stand in for it and how close the correspondence is:

  same_definition   the hardware metric measures the same quantity (e.g. registers per thread, grid
                    size, average active warps / max warps). Differences are real disagreements.
  approximate       same intent, different accounting (e.g. L2 accesses vs L2 sectors, PTX vs SASS
                    instruction counts). Directions are comparable; magnitudes are not.
  proxy             a related but different quantity (e.g. scoreboard-stall share of scheduler cycles
                    vs long-scoreboard stalls per active warp). Treated as weaker evidence.

A hardware value that could not be collected (no Nsight Compute, no counter permission, metric not
available on this GPU or ncu version) is UNOBSERVABLE, never a mismatch.

Profiling is optional: without ncu, validate_hw.py still does timing validation, and registers per
thread are still observable from the native ptxas build (static, recorded as such).
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess

UNOBSERVABLE = "UNOBSERVABLE"

# sim counter -> hardware source. combine: how per-kernel values of a multi-kernel point (v1, v2) are
# combined: sum (counts), dominant (value of the longest kernel), weighted (duration-weighted mean).
MAPPING = {
    "cycles": dict(metrics=["gpu__time_duration.sum"], combine="sum", equivalence="performance",
                   label="Cycles / time",
                   note="simulated cycles vs measured kernel time; compared as a performance direction"),
    "regs": dict(metrics=["launch__registers_per_thread"], combine="dominant", equivalence="same_definition",
                 label="Registers per thread",
                 note="simulator: ptxas on compute_75 PTX; hardware: native sm_80+ SASS"),
    "occupancy": dict(metrics=["sm__warps_active.avg.pct_of_peak_sustained_active"], combine="weighted",
                      equivalence="same_definition", label="Achieved occupancy",
                      note="average active warps / maximum warps per SM"),
    "cta_per_sm": dict(metrics=["launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
                                "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks"],
                       combine="dominant_min", equivalence="same_definition", label="Resident blocks per SM",
                       note="theoretical blocks per SM = min of the four launch occupancy limits"),
    "blocks": dict(metrics=["launch__grid_size"], combine="dominant", equivalence="same_definition",
                   label="Blocks (grid size)", note="dominant kernel"),
    "kernel_launches": dict(metrics=[], combine="count", equivalence="same_definition", label="Kernel launches",
                            note="number of kernels profiled for one forward pass"),
    "l2_accesses": dict(metrics=["lts__t_sectors.sum"], combine="sum", equivalence="approximate",
                        label="L2 traffic", note="simulator L2 accesses vs hardware L2 sectors (32 B)"),
    "dram_sectors": dict(metrics=["dram__sectors_read.sum", "dram__sectors_write.sum"], combine="sum",
                         equivalence="approximate", label="DRAM traffic",
                         note="the 14-SM slice has 1/8 of the A100's L2, so hits and misses can differ"),
    "global_load_insn": dict(metrics=["smsp__inst_executed_op_global_ld.sum"], combine="sum",
                             equivalence="approximate", label="Global load instructions",
                             note="PTX loads (simulator) vs SASS warp-level loads"),
    "shmem_insn": dict(metrics=["smsp__inst_executed_op_shared_ld.sum", "smsp__inst_executed_op_shared_st.sum"],
                       combine="sum", equivalence="approximate", label="Shared-memory instructions",
                       note="PTX vs SASS accounting"),
    "warp_insn": dict(metrics=["smsp__inst_executed.sum"], combine="sum", equivalence="approximate",
                      label="Warp instructions", note="PTX (simulator) vs SASS (hardware)"),
    "thread_insn": dict(metrics=["smsp__thread_inst_executed.sum"], combine="sum", equivalence="approximate",
                        label="Thread instructions", note="PTX (simulator) vs SASS (hardware)"),
    "shmem_bank_conflicts": dict(metrics=["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"], combine="sum",
                                 equivalence="approximate", label="Shared-memory bank conflicts",
                                 note="the simulator models WMMA shared-memory access differently"),
    "issue_util": dict(metrics=["sm__inst_issued.avg.pct_of_peak_sustained_active"], combine="weighted",
                       equivalence="proxy", label="Pipeline (issue) utilization",
                       note="simulator: warp insns / (cycles x SMs x 4); hardware: issue-slot utilization"),
    "scoreboard_frac": dict(metrics=["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
                            combine="weighted", equivalence="proxy", label="Dependency (scoreboard) stalls",
                            note="simulator: share of scheduler cycles with no issue due to scoreboard; "
                                 "hardware: long-scoreboard stalls per active warp"),
}
# Not mapped (no hardware counterpart used): w0_idle, l2_misses, global_store_insn, smem_conflict_rate
# and other derived ratios are recomputed from mapped values when the policy is rerun on hardware.

PERM_ERRORS = ("ERR_NVGPUCTRPERM", "does not have permission", "permission to access NVIDIA GPU Performance Counters")
_BAD_METRIC = re.compile(r"Failed to find metric (\S+)")


def ncu_path() -> str | None:
    return shutil.which("ncu")


def all_metrics() -> list[str]:
    out = []
    for m in MAPPING.values():
        out += m["metrics"]
    return sorted(set(out))


def _num(v: str):
    v = (v or "").replace(",", "").strip()
    if v in ("", "n/a", "N/A"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_ncu_csv(text: str) -> list[dict]:
    """Parse `ncu --csv --page raw` output: one dict per profiled kernel (metric -> float/None)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header = rows[0]
    out = []
    for r in rows[1:]:
        if len(r) != len(header) or not r[0].strip().isdigit():
            continue  # units row or noise
        d = dict(zip(header, r))
        k = {"kernel": d.get("Kernel Name", "")}
        for m in all_metrics():
            k[m] = _num(d[m]) if m in d else None
        out.append(k)
    return out


def profile(exe: str, spec: str, BH: int, N: int, causal: int, metrics: list[str] | None = None,
            timeout: int = 900) -> dict:
    """Profile one design point. Returns {'status', 'kernels': [...], 'dropped_metrics': [...], 'log'}.

    status: ok | no_ncu | no_permission | error. Metrics this ncu does not know are dropped (and become
    UNOBSERVABLE) instead of failing the run."""
    ncu = ncu_path()
    if not ncu:
        return {"status": "no_ncu", "kernels": [], "dropped_metrics": [], "log": ""}
    metrics = list(metrics or all_metrics())
    dropped = []
    env = {k: v for k, v in os.environ.items() if k != "HARNESS_ITERS"}  # one launch of each kernel
    for _ in range(len(metrics) + 1):
        cmd = [ncu, "--csv", "--page", "raw", "--print-units", "base", "--target-processes", "all",
               "--metrics", ",".join(metrics), exe, spec, str(BH), str(N), str(causal)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout)
        log = p.stdout
        if any(e in log for e in PERM_ERRORS):
            return {"status": "no_permission", "kernels": [], "dropped_metrics": dropped, "log": log[-2000:]}
        bad = _BAD_METRIC.findall(log)
        if bad:
            for b in bad:
                b = b.strip(".'\"")
                if b in metrics:
                    metrics.remove(b)
                    dropped.append(b)
            if metrics:
                continue
        kernels = parse_ncu_csv(log)
        return {"status": "ok" if kernels else "error", "kernels": kernels, "dropped_metrics": dropped,
                "log": "" if kernels else log[-2000:], "raw": log}
    return {"status": "error", "kernels": [], "dropped_metrics": dropped, "log": ""}


def aggregate(kernels: list[dict]) -> dict:
    """Combine per-kernel metrics into one value per mapped simulator counter (None = UNOBSERVABLE)."""
    if not kernels:
        return {k: None for k in MAPPING}
    dur = [k.get("gpu__time_duration.sum") or 0.0 for k in kernels]
    dom = kernels[max(range(len(kernels)), key=lambda i: dur[i])]
    out = {}
    for key, m in MAPPING.items():
        c = m["combine"]
        if c == "count":
            out[key] = float(len(kernels))
            continue
        vals = [[k.get(x) for x in m["metrics"]] for k in kernels]
        if c == "sum":
            if any(v is None for row in vals for v in row):
                out[key] = None
            else:
                out[key] = sum(sum(row) for row in vals)
        elif c == "dominant":
            out[key] = dom.get(m["metrics"][0])
        elif c == "dominant_min":
            lim = [dom.get(x) for x in m["metrics"]]
            out[key] = min(lim) if all(v is not None for v in lim) else None
        elif c == "weighted":
            xs = [k.get(m["metrics"][0]) for k in kernels]
            if any(x is None for x in xs) or sum(dur) == 0:
                out[key] = None
            else:
                out[key] = sum(x * d for x, d in zip(xs, dur)) / sum(dur)
    # occupancy limiter of the dominant kernel, in the simulator's words
    lims = {"regs": dom.get("launch__occupancy_limit_registers"), "shmem": dom.get("launch__occupancy_limit_shared_mem"),
            "warps": dom.get("launch__occupancy_limit_warps"), "blocks": dom.get("launch__occupancy_limit_blocks")}
    known = {k: v for k, v in lims.items() if v is not None}
    out["cta_limiter"] = min(known, key=known.get) if len(known) == 4 else None
    return out
