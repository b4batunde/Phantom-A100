"""Architectural counters from one GPGPU-Sim run, plus the derived metrics the rules read.

Raw counters come straight from the simulator log (the same log sim/run_sim.py parses). Derived
metrics normalize them by the workload so they are comparable across kernels:

    l2_traffic_ratio      L2 sector accesses / compulsory sectors (read Q, K, V, write O once).
                          1-2 = near-minimal traffic; >> 1 = redundant or uncoalesced traffic,
                          or an intermediate (the N x N scores) going through memory.
    insn_per_flop         thread instructions / attention FLOPs. Scalar FMA math is ~0.5-1;
                          tensor-core math is far below 0.5 (one mma covers 4096 FLOPs).
    issue_util            warp instructions / (cycles x SMs x 4 schedulers).
    scoreboard_frac       share of scheduler cycles where no warp could issue because of a
                          pending dependency (memory or long-latency op): latency-bound signal.
    smem_conflict_rate    shared-memory bank conflicts per shared-memory instruction.
    waves                 blocks / (SMs x resident blocks per SM).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .space import DesignPoint, Workload

SLICE_SMS = 14          # sim/configs/A100_slice
SCHEDULERS_PER_SM = 4

_RAW = {  # last occurrence wins: cumulative totals after the final kernel launch
    "cycles": (r"^gpu_tot_sim_cycle = (\d+)", int),
    "thread_insn": (r"^gpu_tot_sim_insn = (\d+)", int),
    "warp_insn": (r"^gpgpu_n_tot_w_icount = (\d+)", int),
    "occupancy": (r"^gpu_tot_occupancy = ([\d.]+)%", float),
    "l2_accesses": (r"^L2_total_cache_accesses = (\d+)", int),
    "l2_misses": (r"^L2_total_cache_misses = (\d+)", int),
    "shmem_insn": (r"^gpgpu_n_shmem_insn = (\d+)", int),
    "global_load_insn": (r"^gpgpu_n_load_insn\s+= (\d+)", int),
    "global_store_insn": (r"^gpgpu_n_store_insn = (\d+)", int),
    "dram_reads": (r"^total dram reads = (\d+)", int),
    "dram_writes": (r"^total dram writes = (\d+)", int),
    "shmem_bank_conflicts": (r"^gpgpu_n_shmem_bkconflict = (\d+)", int),
    "w0_idle": (r"W0_Idle:(\d+)", int),
    "w0_scoreboard": (r"W0_Scoreboard:(\d+)", int),
}
_KERNEL_RES = re.compile(r"Kernel '(\S+)' : regs=(\d+), lmem=(\d+), smem=(\d+)")
_LAUNCH = re.compile(r"^kernel_name = (\S+)", re.M)
_SIM_CYCLE = re.compile(r"^gpu_sim_cycle = (\d+)", re.M)
_CTA = re.compile(r"CTA/core = (\d+), limited by: ([\w ]+?)\s*$", re.M)
_RESULT = re.compile(r"RESULT kernel=(\S+) BH=(\d+) N=(\d+) causal=(\d) max_abs_err=(\S+) status=(\w+)")
_ISSUE_DIST = re.compile(r"^Stall:\d+\s+(.*)$", re.M)

# Counters a transformation signature may refer to (raw or derived).
SIGNATURE_COUNTERS = ("cycles", "thread_insn", "warp_insn", "occupancy", "regs", "lmem", "l2_accesses",
                      "dram_sectors", "shmem_insn", "global_load_insn", "shmem_bank_conflicts",
                      "kernel_launches", "scoreboard_frac", "cta_per_sm")


@dataclass
class Counters:
    status: str
    max_abs_err: float
    raw: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)

    def __getitem__(self, k):
        if k in self.derived:
            return self.derived[k]
        return self.raw[k]

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        return {"status": self.status, "max_abs_err": self.max_abs_err, **self.raw, **self.derived}

    @staticmethod
    def from_dict(d: dict) -> "Counters":
        d = dict(d)
        status, err = d.pop("status"), d.pop("max_abs_err")
        derived_keys = {"l2_traffic_ratio", "insn_per_flop", "issue_util", "scoreboard_frac",
                        "smem_conflict_rate", "waves", "dram_sectors", "blocks"}
        der = {k: v for k, v in d.items() if k in derived_keys}
        raw = {k: v for k, v in d.items() if k not in derived_keys}
        return Counters(status, err, raw, der)


def parse_log(log: str, point: DesignPoint, wl: Workload) -> Counters:
    raw: dict = {}
    for k, (rx, conv) in _RAW.items():
        hits = re.findall(rx, log, flags=re.M)
        raw[k] = conv(hits[-1]) if hits else 0

    m = _RESULT.findall(log)
    status, err = (m[-1][5], float(m[-1][4])) if m else ("NO_RESULT", float("nan"))

    # Per-launch information; the dominant kernel is the launch with the most simulated cycles.
    resources = {name: (int(r), int(l), int(s)) for name, r, l, s in _KERNEL_RES.findall(log)}
    launches = _LAUNCH.findall(log)
    per_cycles = [int(c) for c in _SIM_CYCLE.findall(log)]
    ctas = _CTA.findall(log)
    raw["kernel_launches"] = len(launches)
    if launches:
        idx = max(range(len(launches)), key=lambda i: per_cycles[i] if i < len(per_cycles) else 0)
        dom = launches[idx]
        raw["dominant_kernel"] = dom
        raw["regs"], raw["lmem"], _ = resources.get(dom, (0, 0, 0))
        raw["regs_max"] = max(resources.get(n, (0, 0, 0))[0] for n in launches)
        raw["lmem"] = max(resources.get(n, (0, 0, 0))[1] for n in launches)
        if idx < len(ctas):
            raw["cta_per_sm"], raw["cta_limiter"] = int(ctas[idx][0]), ctas[idx][1].strip()
    raw.setdefault("cta_per_sm", 0)
    raw.setdefault("cta_limiter", "")

    issue = _ISSUE_DIST.findall(log)
    issued_cycles = 0
    if issue:
        for tok in issue[-1].split():
            name, _, val = tok.partition(":")
            if name.startswith("W") and not name.startswith("W0"):
                issued_cycles += int(val)

    der: dict = {}
    der["dram_sectors"] = raw["dram_reads"] + raw["dram_writes"]
    der["l2_traffic_ratio"] = raw["l2_accesses"] / wl.compulsory_sectors()
    der["insn_per_flop"] = raw["thread_insn"] / wl.flops()
    cyc = max(raw["cycles"], 1)
    der["issue_util"] = raw["warp_insn"] / (cyc * SLICE_SMS * SCHEDULERS_PER_SM)
    sched = raw["w0_idle"] + raw["w0_scoreboard"] + issued_cycles
    der["scoreboard_frac"] = raw["w0_scoreboard"] / sched if sched else 0.0
    der["smem_conflict_rate"] = raw["shmem_bank_conflicts"] / raw["shmem_insn"] if raw["shmem_insn"] else 0.0
    blocks = wl.blocks(point)
    der["blocks"] = blocks
    der["waves"] = blocks / (SLICE_SMS * raw["cta_per_sm"]) if blocks and raw["cta_per_sm"] else 0.0
    return Counters(status, err, raw, der)


def relative_delta(before: Counters, after: Counters, k: str) -> float:
    """Relative change of counter k; +inf / -inf when it appears from / drops to zero."""
    a, b = before.get(k, 0) or 0, after.get(k, 0) or 0
    if a == 0:
        return 0.0 if b == 0 else float("inf")
    return (b - a) / abs(a)
