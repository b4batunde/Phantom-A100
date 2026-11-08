"""Simulate every sim-verifiable design point once (the ground truth that searches are scored on).

    source <gpgpu-sim env>; python -m autotune.groundtruth --jobs 2 [--BH 42 --N 128 --causal 0]
           [--import-v1v2 path/to/original/perf/logs]

Points whose performance the simulator cannot judge (double buffering) are not simulated.
"""
from __future__ import annotations

import argparse
import os

from .backends import GpgpuSimBackend, StaticInfo
from .space import Workload, all_points, make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--BH", type=int, default=42)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--import-v1v2", default=None,
                    help="directory with naive_<BH>_<N>_<c>.log / tiled_... from the original study "
                         "(v1/v2 kernel code is unchanged, so their simulated counters are reused)")
    ap.add_argument("--only", nargs="*", default=None, help="point keys to run (default: all)")
    a = ap.parse_args()
    wl = Workload(a.BH, a.N, a.causal)
    static = StaticInfo.from_ptxas()
    be = GpgpuSimBackend(static=static)
    if a.import_v1v2:
        for name in ("naive", "tiled"):
            f = os.path.join(a.import_v1v2, f"{name}_{wl.BH}_{wl.N}_{wl.causal}.log")
            if os.path.exists(f) and be.cache.get(make(name), wl) is None:
                be.import_log(make(name), wl, f, "simulated (original v1-v4 study log; kernel code unchanged)")
                print(f"imported {f}")
    pts = all_points(include_blind=False)
    if a.only:
        pts = [p for p in pts if p.key() in a.only]
    # longest first so the two workers finish together
    order = {"naive": 0, "tiled": 1, "fused": 2, "fused_vec": 3, "fused_tc": 4}
    pts.sort(key=lambda p: order[p.structure])
    print(f"{len(pts)} sim-verifiable points at {wl.key()}; "
          f"{sum(be.cache.get(p, wl) is None for p in pts)} to simulate")
    be.evaluate_many(pts, wl, jobs=a.jobs)
    print("done")


if __name__ == "__main__":
    main()
