"""The transformation menu. Each entry is a named, reversible code change with:

  targets    the bottlenecks (diagnose.py) it is meant to relieve
  expected   its declared counter signature: the direction each counter should move
             ('+' up, '-' down, '0' roughly unchanged, '?' no claim)
  applies    which design points it can be applied to; apply() returns the new point

The signature is written BEFORE any run, from what the code change does mechanically. After a
simulation the policy compares observed counter deltas with it (policy.py). A transformation can
therefore be right about its mechanism but useless (counters move as declared, cycles do not), or
help for a reason it did not predict. Both outcomes are recorded.

Several entries are decoys in the sense that nothing guarantees they help: register caps, tile
shapes and double buffering. They are real optimizations that sometimes pay off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .space import DesignPoint, make, with_


@dataclass(frozen=True)
class Transformation:
    name: str
    description: str
    targets: tuple[str, ...]
    expected: dict = field(default_factory=dict)
    applies: Callable[[DesignPoint], bool] = lambda p: False
    apply: Callable[[DesignPoint], DesignPoint] = lambda p: p
    caveat: str = ""

    def __call__(self, p: DesignPoint) -> DesignPoint | None:
        if not self.applies(p):
            return None
        q = self.apply(p)
        return q if q.is_valid() and q != p else None


def _is(*structs):
    return lambda p: p.structure in structs


MENU: list[Transformation] = [
    Transformation(
        "tile_smem", "stage 32x32 operand tiles in shared memory for both matmuls (v1 -> v2 style)",
        targets=("redundant_global_traffic",),
        expected={"global_load_insn": "-", "l2_accesses": "-", "shmem_insn": "+", "kernel_launches": "0",
                  "cycles": "-"},
        applies=_is("naive"), apply=lambda p: make("tiled")),
    Transformation(
        "fuse", "fuse scores, softmax and PV into one kernel with online softmax; scores stay on chip",
        targets=("materialized_intermediate", "redundant_global_traffic"),
        expected={"kernel_launches": "-", "l2_accesses": "-", "dram_sectors": "-", "regs": "+",
                  "occupancy": "-", "cycles": "-"},
        applies=_is("naive", "tiled"), apply=lambda p: make("fused")),
    Transformation(
        "vectorize_fp16", "128-bit global loads and FP16 K/V tiles in shared memory",
        targets=("redundant_global_traffic", "register_limited_occupancy"),
        expected={"global_load_insn": "-", "l2_accesses": "-", "regs": "-", "occupancy": "+", "cycles": "-"},
        applies=_is("fused"),
        apply=lambda p: make("fused_vec", br=p.br, bc=32, maxreg=p.maxreg)),
    Transformation(
        "tensor_cores", "run QK^T and PV on tensor cores (WMMA m16n16k16, FP32 accumulate)",
        targets=("cuda_core_math_bound",),
        expected={"thread_insn": "-", "warp_insn": "-", "regs": "-", "cycles": "-"},
        applies=lambda p: p.structure in ("fused", "fused_vec") and not p.db,
        apply=lambda p: make("fused_tc")),
    Transformation(
        "reg_cap_128", "__launch_bounds__ cap of 128 registers/thread to fit more blocks per SM",
        targets=("register_limited_occupancy", "latency_bound"),
        expected={"regs": "-", "occupancy": "+", "cta_per_sm": "+", "cycles": "-"},
        applies=lambda p: p.structure in ("fused", "fused_vec") and p.maxreg == 0,
        apply=lambda p: with_(p, maxreg=128),
        caveat="if ptxas spills, the simulator does not charge for it (it executes PTX, which has no spills)"),
    Transformation(
        "remove_reg_cap", "drop the register cap",
        targets=("register_spills",),
        expected={"regs": "+", "lmem": "-", "occupancy": "-"},
        applies=lambda p: p.maxreg == 128, apply=lambda p: with_(p, maxreg=0)),
    Transformation(
        "grow_br", "128 query rows per block instead of 64: each K/V tile is reused by twice as many rows",
        targets=("redundant_global_traffic",),
        expected={"l2_accesses": "-", "blocks": "-", "cycles": "-"},
        applies=lambda p: p.structure in ("fused", "fused_vec") and p.br == 64,
        apply=lambda p: with_(p, br=128)),
    Transformation(
        "shrink_br", "64 query rows per block instead of 128: twice as many blocks",
        targets=("low_parallelism",),
        expected={"blocks": "+", "occupancy": "+", "cycles": "-"},
        applies=lambda p: p.structure in ("fused", "fused_vec") and p.br == 128,
        apply=lambda p: with_(p, br=64)),
    Transformation(
        "grow_bc", "64 keys per K/V tile instead of 32: half as many tile loads and barriers",
        targets=("latency_bound",),
        expected={"shmem_insn": "?", "cta_per_sm": "?", "cycles": "-"},
        applies=lambda p: p.structure == "fused_vec" and p.bc == 32,
        apply=lambda p: with_(p, bc=64)),
    Transformation(
        "shrink_bc", "32 keys per K/V tile instead of 64: less shared memory per block",
        targets=("smem_limited_occupancy",),
        expected={"cta_per_sm": "+", "occupancy": "+", "cycles": "-"},
        applies=lambda p: p.structure == "fused_vec" and p.bc == 64,
        apply=lambda p: with_(p, bc=32)),
    Transformation(
        "double_buffer", "prefetch the next K/V tile with cp.async while computing the current one",
        targets=("latency_bound",),
        expected={"scoreboard_frac": "-", "cycles": "-"},
        applies=lambda p: p.structure == "fused_vec" and not p.db,
        apply=lambda p: with_(p, db=1),
        caveat="GPGPU-Sim does not implement cp.async: only a real GPU can judge this one"),
]

BY_NAME = {t.name: t for t in MENU}


def neighbors(p: DesignPoint) -> list[tuple[Transformation, DesignPoint]]:
    out = []
    for t in MENU:
        q = t(p)
        if q is not None:
            out.append((t, q))
    return out
