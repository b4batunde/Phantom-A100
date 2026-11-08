"""Run the counter-guided optimizer live on GPGPU-Sim (no precomputed ground truth needed).

    source <gpgpu-sim env>
    python -m autotune.run --start naive --BH 42 --N 128 --budget 8

Prints the decision trace as it goes and writes results/autotune/run_<workload>_<start>.json.
Simulations already in the cache are reused (the simulator is deterministic).
"""
from __future__ import annotations

import argparse
import json
import os

from .backends import REPO, GpgpuSimBackend, StaticInfo
from .policy import Beliefs
from .search import counter_guided
from .space import DesignPoint, Workload
from .transforms import MENU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="naive", help="design point key, e.g. naive, fused/br64/bc32")
    ap.add_argument("--BH", type=int, default=42)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--beliefs", default=None, help="start from saved beliefs (e.g. results/autotune/beliefs.json)")
    a = ap.parse_args()
    wl = Workload(a.BH, a.N, a.causal)
    be = GpgpuSimBackend(static=StaticInfo.from_ptxas())
    beliefs = Beliefs.load(a.beliefs) if a.beliefs else None
    res, beliefs = counter_guided(DesignPoint.from_key(a.start), wl, be, budget=a.budget, beam=a.beam, beliefs=beliefs)
    for dg in res.diagnoses:
        print(f"\nstep {dg['step']}: {dg['incumbent']} ({dg['cycles']:,} cycles)")
        for b in dg["bottlenecks"]:
            print(f"   - {b}")
        for o in (o for o in res.outcomes if o["step"] == dg["step"]):
            print(f"   tried {o['transform']:15} -> {o['child']:28} {o['speedup']:.2f}x  {o['verdict']:18} "
                  f"agreement {o['agreement']:.2f}{'' if o['trusted'] else '  UNTRUSTED: ' + '; '.join(o['flags'])}")
        if dg.get("unlock"):
            print(f"   unlock: {dg['unlock']}")
    bc, bk = res.best
    print(f"\npath: {' -> '.join(res.path)}\nbest trusted: {bk} ({bc:,} cycles), {len(res.evaluated)} simulations\n"
          f"stop: {res.stop_reason}")
    for h in res.hardware_candidates:
        print(f"hardware-only candidate: {h['transform']} -> {h['point']} ({'; '.join(h['why_not_simulated'])})")
    for t in MENU:
        for k, (want, seen, status) in beliefs.learned_signature(t).items():
            if status in ("wrong", "mostly holds"):
                print(f"signature check: {t.name}.{k} declared {want}, observed {seen} -> {status}")
    out = os.path.join(REPO, "results", "autotune", f"run_{wl.key()}_{a.start.replace('/', '_')}.json")
    json.dump({"result": res.__dict__, "beliefs": beliefs.to_dict()}, open(out, "w"), indent=1, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
