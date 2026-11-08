"""Bottleneck classifier: explicit, printable rules over simulator counters.

Every rule returns a Bottleneck with a severity (how far past its threshold the counter is, >= 1
means triggered) and the evidence string that justified it. There is no learned model here: the
point is that a person can read why the optimizer believed what it believed.

Thresholds were set from first principles (what the counter means), not fitted to the
historical v1 -> v4 path. They are listed in THRESHOLDS so the README can quote them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .counters import Counters
from .space import DesignPoint

THRESHOLDS = {
    "materialized_intermediate": "more than one kernel launch AND L2 traffic > 4x compulsory",
    "redundant_global_traffic": "L2 traffic > 3x compulsory (Q, K, V, O moved once)",
    "register_limited_occupancy": "resident blocks limited by registers AND occupancy < 25%",
    "smem_limited_occupancy": "resident blocks limited by shared memory AND occupancy < 25%",
    "latency_bound": "> 50% of scheduler cycles stalled on dependencies (scoreboard) AND occupancy < 30%",
    "cuda_core_math_bound": "> 0.5 thread instructions per attention FLOP (matmuls run as scalar FMAs)",
    "smem_bank_conflicts": "> 0.05 bank conflicts per shared-memory instruction",
    "low_parallelism": "< 0.75 waves of blocks (SMs left idle)",
    "register_spills": "ptxas for sm_80 reports spill stores/loads for the kernel (static check, not simulated)",
}


@dataclass(frozen=True)
class Bottleneck:
    name: str
    severity: float
    evidence: str

    def __str__(self):
        return f"{self.name} (severity {self.severity:.2f}): {self.evidence}"


def diagnose(c: Counters, p: DesignPoint) -> list[Bottleneck]:
    out: list[Bottleneck] = []
    occ = c["occupancy"]
    l2r = c["l2_traffic_ratio"]

    if c["kernel_launches"] > 1 and l2r > 4:
        out.append(Bottleneck("materialized_intermediate", l2r / 4,
                              f"{c['kernel_launches']} launches, L2 traffic {l2r:.1f}x compulsory"))
    if l2r > 3:
        out.append(Bottleneck("redundant_global_traffic", l2r / 3, f"L2 traffic {l2r:.1f}x compulsory"))
    lim = c.get("cta_limiter", "")
    if "reg" in lim and occ < 25:
        out.append(Bottleneck("register_limited_occupancy", 25 / max(occ, 1),
                              f"{c['cta_per_sm']} blocks/SM limited by registers ({c['regs']} regs/thread), "
                              f"occupancy {occ:.1f}%"))
    if "smem" in lim and occ < 25:
        out.append(Bottleneck("smem_limited_occupancy", 25 / max(occ, 1),
                              f"{c['cta_per_sm']} blocks/SM limited by shared memory, occupancy {occ:.1f}%"))
    sb = c["scoreboard_frac"]
    if sb > 0.5 and occ < 30:
        out.append(Bottleneck("latency_bound", sb / 0.5,
                              f"{100 * sb:.0f}% of scheduler cycles stalled on dependencies, occupancy {occ:.1f}%"))
    ipf = c["insn_per_flop"]
    if ipf > 0.5:
        out.append(Bottleneck("cuda_core_math_bound", ipf / 0.5,
                              f"{ipf:.2f} thread instructions per FLOP (tensor-core kernels are < 0.2)"))
    scr = c["smem_conflict_rate"]
    if scr > 0.05:
        out.append(Bottleneck("smem_bank_conflicts", scr / 0.05, f"{scr:.2f} conflicts per shared-memory instruction"))
    w = c["waves"]
    if 0 < w < 0.75:
        out.append(Bottleneck("low_parallelism", 0.75 / w, f"{w:.2f} waves of blocks"))
    spill = c.get("hw_spill_bytes", 0) or 0
    if spill > 0:
        out.append(Bottleneck("register_spills", 1.0 + spill / 64,
                              f"ptxas (sm_80) spills {spill} bytes/thread; cost invisible to the simulator"))
    out.sort(key=lambda b: -b.severity)
    return out
