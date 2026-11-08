"""The attention design space: every kernel configuration the optimizer can reach.

A DesignPoint is one runnable kernel. Its fields are the knobs that the transformations in
transforms.py turn; `harness_spec()` maps it to the kernel name sim/harness.cu understands, so the
same point runs on GPGPU-Sim and on a real GPU.

    structure  "naive" | "tiled"   three kernels, N x N scores in HBM (v1, v2)
               "fused"             one kernel, online softmax, FP32 shared-memory tiles (v3)
               "fused_vec"         + 128-bit loads, FP16 K/V tiles (v3.1, v3.2 with db=1)
               "fused_tc"          + both matmuls on tensor cores via WMMA (v4)
    br, bc     query rows per block, keys per K/V tile (fused, fused_vec)
    db         double-buffered K/V with cp.async (fused_vec)
    maxreg     0 = no register cap, 128 = __launch_bounds__ cap of 128 registers/thread

The space is deliberately larger than the historical v1 -> v4 path: register caps, tile shapes
and double buffering are all reachable, and most of them do not help. They are the decoys.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

STRUCTURES = ("naive", "tiled", "fused", "fused_vec", "fused_tc")

# Compiled configurations (must match csrc/kernels/*.cuh launchers).
FUSED_BR = (64, 128)                      # fused: BC fixed at 32
VEC_TILES = ((64, 32), (64, 64), (128, 32), (128, 64))
MAXREG = (0, 128)

# Historical hand-optimized versions, for the rediscovery experiment.
HISTORICAL = {
    "v1 naive": dict(structure="naive"),
    "v2 tiled": dict(structure="tiled"),
    "v3 fused": dict(structure="fused", br=64, bc=32),
    "v3.1 fused_vec": dict(structure="fused_vec", br=64, bc=32),
    "v3.2 fused_db": dict(structure="fused_vec", br=64, bc=32, db=1),
    "v4 fused_tc": dict(structure="fused_tc", br=64, bc=64),
}


@dataclass(frozen=True, order=True)
class DesignPoint:
    structure: str
    br: int = 0
    bc: int = 0
    db: int = 0
    maxreg: int = 0

    # ------------------------------------------------------------------ validity
    def is_valid(self) -> bool:
        s = self.structure
        if s in ("naive", "tiled"):
            return (self.br, self.bc, self.db, self.maxreg) == (0, 0, 0, 0)
        if s == "fused":
            return self.br in FUSED_BR and self.bc == 32 and self.db == 0 and self.maxreg in MAXREG
        if s == "fused_vec":
            return (self.br, self.bc) in VEC_TILES and self.db in (0, 1) and self.maxreg in MAXREG
        if s == "fused_tc":
            return (self.br, self.bc, self.db, self.maxreg) == (64, 64, 0, 0)
        return False

    # ------------------------------------------------------------------ what the simulator can judge
    def sim_blind_spots(self) -> list[str]:
        """Features of this point whose performance GPGPU-Sim does not model (known before running)."""
        out = []
        if self.db:
            out.append("cp.async double buffering is not implemented in GPGPU-Sim (the sm_75 PTX path "
                       "falls back to synchronous copies)")
        return out

    def sim_verifiable(self) -> bool:
        return not self.sim_blind_spots()

    # ------------------------------------------------------------------ naming
    def harness_spec(self) -> str:
        s = self.structure
        if s in ("naive", "tiled", "fused_tc"):
            return s
        if s == "fused":
            return "fused" if (self.br, self.maxreg) == (64, 0) else f"fused_cfg:{self.br}:{self.maxreg}"
        return f"fused_tuned:{self.br}:{self.bc}:{self.db}:{self.maxreg}"

    def key(self) -> str:
        if self.structure in ("naive", "tiled", "fused_tc"):
            return self.structure
        k = f"{self.structure}/br{self.br}/bc{self.bc}"
        if self.db:
            k += "/db"
        if self.maxreg:
            k += f"/r{self.maxreg}"
        return k

    def historical_name(self) -> str | None:
        for name, kw in HISTORICAL.items():
            if self == make(**kw):
                return name
        return None

    def label(self) -> str:
        h = self.historical_name()
        return f"{self.key()} ({h})" if h else self.key()

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_key(key: str) -> "DesignPoint":
        parts = key.split("/")
        kw: dict = {"structure": parts[0]}
        if parts[0] == "fused_tc":
            kw.update(br=64, bc=64)
        for p in parts[1:]:
            if p.startswith("br"):
                kw["br"] = int(p[2:])
            elif p.startswith("bc"):
                kw["bc"] = int(p[2:])
            elif p == "db":
                kw["db"] = 1
            elif p.startswith("r"):
                kw["maxreg"] = int(p[1:])
        pt = DesignPoint(**kw)
        assert pt.is_valid(), key
        return pt


def make(structure: str, **kw) -> DesignPoint:
    if structure == "fused_tc":
        kw.setdefault("br", 64)
        kw.setdefault("bc", 64)
    if structure in ("fused", "fused_vec"):
        kw.setdefault("br", 64)
        kw.setdefault("bc", 32)
    p = DesignPoint(structure, **kw)
    assert p.is_valid(), p
    return p


def with_(p: DesignPoint, **kw) -> DesignPoint:
    return replace(p, **kw)


def all_points(include_blind: bool = True) -> list[DesignPoint]:
    pts = [make("naive"), make("tiled"), make("fused_tc")]
    pts += [make("fused", br=br, maxreg=r) for br in FUSED_BR for r in MAXREG]
    pts += [make("fused_vec", br=br, bc=bc, db=db, maxreg=r)
            for br, bc in VEC_TILES for db in (0, 1) for r in MAXREG]
    pts = sorted(set(pts))
    return [p for p in pts if include_blind or p.sim_verifiable()]


@dataclass(frozen=True)
class Workload:
    """One attention problem: BH = batch x heads, N = sequence length, d = 64 (compiled)."""
    BH: int = 42
    N: int = 128
    causal: int = 0
    d: int = 64

    def key(self) -> str:
        return f"BH{self.BH}_N{self.N}_c{self.causal}"

    def flops(self) -> float:
        return 4.0 * self.BH * self.N * self.N * self.d * (0.5 if self.causal else 1.0)

    def compulsory_sectors(self) -> float:
        """32-byte L2 sectors that any kernel must move: read Q, K, V and write O once, FP16."""
        return 4 * self.BH * self.N * self.d * 2 / 32

    def blocks(self, p: DesignPoint) -> int:
        """Thread blocks of the dominant kernel (fused kernels: one per BR query rows per head)."""
        if p.structure in ("naive", "tiled"):
            return 0  # several kernels; not used by the rules
        return -(-self.N // p.br) * self.BH
