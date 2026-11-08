"""Freeze autotune-v1 and pre-register its predictions before any hardware measurement.

    python -m autotune.freeze write   [--commit autotune-v1-sim-only]   # manifest + predictions
    python -m autotune.freeze verify                                    # does the tree still match?

Writes results/autotune/preregistration/{manifest.json, predictions.json, README.md}.

manifest.json     the frozen commit and the SHA-256 of every file that defines the experimental
                  subject (optimizer code, kernels, harness, simulator config and patch, simulated data).
                  Hashes are computed from the frozen commit itself (git blobs), not the working tree.
predictions.json  everything the frozen optimizer and simulator claim, in a form validate_hw.py scores
                  mechanically: the full simulated ranking, the search path and its decisions, untrusted
                  and unverifiable candidates, per-transition predicted counter and performance
                  directions, known simulator limitations, and hypotheses with explicit criteria.

Rule: nothing under FROZEN changes after this. Improvements informed by hardware go into autotune-v2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

from .backends import REPO

PREREG = os.path.join(REPO, "results", "autotune", "preregistration")

FROZEN = {
    "optimizer": ["autotune/space.py", "autotune/counters.py", "autotune/diagnose.py", "autotune/transforms.py",
                  "autotune/policy.py", "autotune/search.py", "autotune/backends.py"],
    "experiment_drivers": ["autotune/groundtruth.py", "autotune/experiments.py", "autotune/run.py"],
    "kernels_and_harness": ["csrc/kernels/device_common.cuh", "csrc/kernels/naive.cuh", "csrc/kernels/tiled.cuh",
                            "csrc/kernels/fused.cuh", "csrc/kernels/fused_tuned.cuh", "csrc/kernels/fused_tc.cuh",
                            "sim/harness.cu"],
    "simulator": ["sim/configs/A100_slice/gpgpusim.config",
                  "sim/patches/all-fixes-for-gpgpu-sim-dev-4195dd4.patch"],
    "simulated_data": ["results/autotune/sim_cache.json", "results/autotune/experiments.json",
                       "results/autotune/static_sm80.json", "results/autotune/beliefs.json",
                       "results/autotune/simulator_logs.tar.gz"],
}

# Tolerances, fixed here so validation cannot pick them after seeing data.
COUNTER_TOL = 0.03   # a counter moved if |relative change| > 3%   (same as policy.SIGN_TOL)
PERF_TOL = 0.02      # faster/slower if speedup outside [0.98, 1.02] (same as policy.MIN_GAIN)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, stdout=subprocess.PIPE, check=True).stdout.decode().strip()


def blob_sha256(commit: str, path: str) -> str:
    data = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, stdout=subprocess.PIPE, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str) -> str:
    return hashlib.sha256(open(os.path.join(REPO, path), "rb").read()).hexdigest()


def sign(rel: float, tol: float) -> str:
    if rel != rel:
        return "?"
    return "+" if rel > tol else "-" if rel < -tol else "0"


def perf_dir(speedup: float) -> str:
    return "faster" if speedup >= 1 + PERF_TOL else "slower" if speedup <= 1 - PERF_TOL else "same"


# ------------------------------------------------------------------------------------- manifest
def manifest(commit: str) -> dict:
    full = git("rev-parse", f"{commit}^{{commit}}")
    files = {grp: {p: blob_sha256(full, p) for p in paths} for grp, paths in FROZEN.items()}
    return {
        "subject": "autotune-v1",
        "tag": commit if commit != full else None,
        "frozen_commit": full,
        "frozen_commit_date": git("show", "-s", "--format=%cI", full),
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hardware_measurements_observed": False,
        "statement": "All optimization rules, counter signatures, confidence logic and the search algorithm "
                     "were specified, and all predictions below were written, before any A100 measurement "
                     "was observed. Nothing in `files` may change; hardware-informed changes go to autotune-v2.",
        "sha256": files,
        "tolerances": {"counter_direction": COUNTER_TOL, "performance_direction": PERF_TOL},
    }


def verify(man: dict) -> list[str]:
    bad = []
    for grp, files in man["sha256"].items():
        for p, h in files.items():
            if not os.path.exists(os.path.join(REPO, p)):
                bad.append(f"missing: {p}")
            elif file_sha256(p) != h:
                bad.append(f"changed since freeze: {p}")
    return bad


# ------------------------------------------------------------------------------------- predictions
def predictions(man: dict) -> dict:
    from .backends import ReplayBackend
    from .counters import relative_delta
    from .hw_counters import MAPPING
    from .policy import trust_flags
    from .space import DesignPoint, Workload, all_points
    from .transforms import MENU, neighbors

    def sim_dir(c0, c1, k):
        # the simulator side has no grid size for the three-kernel v1/v2 (counters.Workload.blocks): no prediction
        if k == "blocks" and (not c0.get(k) or not c1.get(k)):
            return "?"
        return sign(relative_delta(c0, c1, k), COUNTER_TOL)

    exp = json.load(open(os.path.join(REPO, "results", "autotune", "experiments.json")))
    wl = Workload()
    be = ReplayBackend()
    sim = {p.key(): be.evaluate(p, wl) for p in all_points(include_blind=False)}

    ranking = []
    for i, r in enumerate(exp["ground_truth"], 1):
        p_key = r["point"]
        c = sim[p_key]
        ranking.append({"rank": i, "point": p_key, "label": r["label"], "sim_cycles": r["cycles"],
                        "trusted": r["trusted"], "flags": trust_flags(c, DesignPoint.from_key(p_key)),
                        "sim_regs": c["regs"], "sm80_regs": c.get("hw_regs", 0),
                        "sm80_spill_bytes": c.get("hw_spill_bytes", 0),
                        "sim_occupancy_pct": round(c["occupancy"], 2), "sim_blocks_per_sm": c["cta_per_sm"]})

    # Every transition the menu allows between design points, with the simulator's predicted
    # direction for each mapped counter and for performance, plus the rule's declared signature.
    transitions = []
    for p in all_points(include_blind=True):
        for t, q in neighbors(p):
            row = {"transform": t.name, "parent": p.key(), "child": q.key(),
                   "declared_signature": dict(t.expected), "targets": list(t.targets)}
            if p.key() in sim and q.key() in sim:
                c0, c1 = sim[p.key()], sim[q.key()]
                sp = c0["cycles"] / c1["cycles"]
                row.update(sim_verifiable=True, sim_speedup=round(sp, 4), predicted_performance=perf_dir(sp),
                           child_trusted=not trust_flags(c1, q),
                           sim_counter_direction={k: sim_dir(c0, c1, k) for k in MAPPING if k != "cycles"})
            else:
                row.update(sim_verifiable=False, sim_speedup=None,
                           predicted_performance=("faster" if t.expected.get("cycles") == "-" else "unknown"),
                           prediction_source="declared signature only (simulator cannot judge this point)")
            transitions.append(row)

    red = exp["rediscovery"]
    path_steps = list(zip(red["path"][:-1], red["path"][1:]))

    hypotheses = [
        {"id": "H1", "question": "performance fidelity",
         "claim": "Simulated cycles rank the 15 sim-verifiable points in nearly the same order as measured A100 time.",
         "metric": "spearman_rho(sim_cycles, measured_ms) over the 15 sim-verifiable points",
         "criterion": {"supported": ">= 0.8", "partial": ">= 0.5 and < 0.8", "refuted": "< 0.5"}},
        {"id": "H2", "question": "performance fidelity",
         "claim": "v4 fused_tc, the simulator's best trusted point, is the fastest of all 23 points on the A100.",
         "metric": "argmin(measured_ms) over all 23 points == 'fused_tc'",
         "criterion": {"supported": "true", "refuted": "false"}},
        {"id": "H3", "question": "performance fidelity (v2/v3 ordering)",
         "claim": "Simulator ordering: v2 tiled is faster than v3 fused (simulated v3/v2 time ratio "
                  f"{sim['fused/br64/bc32']['cycles'] / sim['tiled']['cycles']:.2f}).",
         "risk_identified_before_measurement":
             "The simulator's occupancy model gives v3 242 registers/thread (ptxas on the compute_75 PTX it "
             "executes: 4 blocks/SM), while the native sm_80 build uses 168 (6 blocks/SM). If register-limited "
             "occupancy is what makes v3 slow in simulation, v3 may be close to or faster than v2 on the A100. "
             "If so, the unlock move that carried the search from v2 to v3 was needed only because of a "
             "simulator artifact, and pure greedy search on hardware counters would not have stopped at v2.",
         "metric": "measured_ms(fused/br64/bc32) / measured_ms(tiled)",
         "criterion": {"simulator ordering holds": "> 1.02", "tie": "0.98 to 1.02", "reversed": "< 0.98"}},
        {"id": "H4", "question": "trust model (register spills)",
         "claim": "Simulated gains of register-capped (spilling) points overstate hardware gains: that is why "
                  "they were marked untrusted.",
         "metric": "for each of the 6 capped/uncapped pairs: measured speedup of capping < simulated speedup "
                   "of capping minus 0.02",
         "criterion": {"supported": ">= 4 of 6 pairs", "refuted": "<= 2 of 6 pairs", "inconclusive": "3 of 6"}},
        {"id": "H5", "question": "decision fidelity",
         "claim": "Every accepted step on the guided search path is in the same performance direction on the "
                  "A100 (improving steps faster, the unlock step slower).",
         "metric": "per step on the path: perf_dir(measured parent/child) == predicted_performance",
         "steps": [{"parent": a, "child": b} for a, b in path_steps],
         "criterion": {"supported": "all steps match", "partial": "all improving steps match",
                       "refuted": "an improving step is slower on hardware"}},
        {"id": "H6", "question": "decision fidelity",
         "claim": "The kernel the frozen optimizer selected (fused_tc) is within 5% of the measured-fastest "
                  "point among all 23.",
         "metric": "measured_ms(fused_tc) / min(measured_ms) <= 1.05",
         "criterion": {"supported": "true", "refuted": "false"}},
        {"id": "H7", "question": "mechanism fidelity",
         "claim": "Counters whose hardware metric has the same definition as the simulator counter predict "
                  "direction better than counters mapped to an approximate or proxy metric.",
         "metric": "direction accuracy of simulator-predicted counter changes, grouped by mapping equivalence "
                   "(same_definition / approximate / proxy), over all sim-verifiable transitions",
         "expectation": "same_definition >= 0.9; approximate and proxy lower. Stated as an expectation, not a "
                        "pass/fail threshold.",
         "criterion": {"supported": "accuracy(same_definition) > accuracy(approximate) and > accuracy(proxy)",
                       "refuted": "otherwise"}},
        {"id": "H8", "question": "rule-only prediction (no simulator evidence)",
         "claim": "double_buffer (cp.async) makes fused_vec faster on the A100, as its declared signature says. "
                  "The simulator could not test this.",
         "metric": "for each of the 4 tile configs without a register cap: measured speedup of db=1 over db=0",
         "criterion": {"supported": ">= 3 of 4 faster by > 2%", "refuted": "<= 1 of 4 faster by > 2%",
                       "inconclusive": "2 of 4"}},
        {"id": "H9", "question": "decision fidelity (search on hardware counters)",
         "claim": "Run unchanged on hardware counters (Nsight Compute mapped through the frozen mapping) and "
                  "hardware time, the frozen policy also ends at fused_tc.",
         "metric": "final incumbent of counter_guided(start=naive) with HardwareBackend",
         "criterion": {"supported": "ends at fused_tc", "refuted": "ends elsewhere",
                       "inconclusive": "Nsight counters unavailable"}},
    ]

    return {
        "subject": "autotune-v1",
        "frozen_commit": man["frozen_commit"],
        "manifest_sha256": hashlib.sha256(json.dumps(man["sha256"], sort_keys=True).encode()).hexdigest(),
        "written_at": man["written_at"],
        "hardware_measurements_observed": False,
        "simulated_workload": {"BH": wl.BH, "N": wl.N, "causal": wl.causal, "d": wl.d,
                               "simulator": "GPGPU-Sim dev@4195dd4 + sim/patches, SM80_A100 config, 14-SM slice"},
        "hardware_workload": {"BH": "round(42 * SMs / 14) (324 on a 108-SM A100)", "N": wl.N, "causal": wl.causal,
                              "why": "same blocks per SM as the 14-SM simulated slice"},
        "tolerances": man["tolerances"],
        "simulated_ranking": ranking,
        "search": {"start": "naive", "path": red["path"], "simulations_used": len(red["evaluated"]),
                   "stop_reason": red["stop_reason"], "decisions": red["outcomes"],
                   "unlock_moves": [d.get("unlock") for d in red["diagnoses"] if d.get("unlock")],
                   "diagnoses": red["diagnoses"],
                   "greedy_ablation_path": exp["ablation_no_unlock"]["path"]},
        "untrusted_candidates": [r for r in ranking if not r["trusted"]],
        "unverifiable_candidates": red["hardware_candidates"] + [
            {"point": t["child"], "from": t["parent"], "transform": t["transform"]}
            for t in transitions if not t["sim_verifiable"] and t["transform"] == "double_buffer"
            and t["parent"] != "fused_vec/br64/bc32"],
        "declared_signatures": {t.name: dict(t.expected) for t in MENU},
        "counter_mapping": {k: {kk: vv for kk, vv in m.items()} for k, m in MAPPING.items()},
        "scoring": {
            "counter_match": "direction of hardware change (tolerance above) equals the predicted direction; "
                             "'?' predictions are NO_CLAIM; a missing hardware value is UNOBSERVABLE, not a mismatch",
            "two_predictors": "each transition is scored twice: the simulator's observed direction (does the "
                              "simulator predict hardware?) and the transformation's declared signature (does "
                              "the rule predict hardware?)",
            "questions": {"performance_fidelity": "does simulation rank configurations correctly?",
                          "mechanism_fidelity": "do predicted counter changes occur on hardware?",
                          "decision_fidelity": "would the frozen optimizer have made useful decisions on "
                                               "hardware measurements?"}},
        "transitions": transitions,
        "known_simulator_limitations": [
            "Executes PTX, not SASS: no ptxas scheduling, no register spills. Register-capped points spill "
            "168-188 bytes/thread on sm_80 at no simulated cost.",
            "cp.async is not implemented; double-buffered points run a synchronous fallback and are not simulated.",
            "Register counts for occupancy come from ptxas on the compute_75 PTX the simulator runs; they can "
            "differ from the native sm_80 build (v3: 242 simulated vs 168 sm_80).",
            "14-SM slice with 5 of 40 memory channels: L2 capacity is 1/8 of the real chip, so absolute L2/DRAM "
            "traffic and anything that depends on L2 hits can differ on the full A100.",
            "Cycle counts are not calibrated against hardware; ratios between versions are the intended output.",
            "naive and tiled counters come from the original v1-v4 study's logs (same kernel code).",
        ],
        "hypotheses": hypotheses,
    }


def write_readme(man, pred):
    L = ["# Pre-registration: autotune-v1 (written before any A100 measurement)", "",
         f"- Frozen commit: `{man['frozen_commit']}` (tag `{man['tag']}`), committed {man['frozen_commit_date']}",
         f"- Written: {man['written_at']}",
         f"- Manifest hash: `{pred['manifest_sha256']}`",
         "- Files: `manifest.json` (SHA-256 of every frozen file), `predictions.json` (everything below, machine-readable)",
         "", man["statement"], "",
         "Check the tree still matches: `python -m autotune.freeze verify`. `validate_hw.py` runs the same check and "
         "prints the result at the top of the hardware report.", "",
         "## Hypotheses", ""]
    for h in pred["hypotheses"]:
        L.append(f"**{h['id']} ({h['question']}).** {h['claim']}")
        if "risk_identified_before_measurement" in h:
            L.append(f"  Risk identified in advance: {h['risk_identified_before_measurement']}")
        L.append(f"  Metric: {h['metric']}. Criterion: " + "; ".join(f"{k}: {v}" for k, v in h["criterion"].items()))
        L.append("")
    L += ["## Simulated ranking being tested", "", "| Rank | Point | Simulated cycles | Trusted |", "|---|---|---|---|"]
    for r in pred["simulated_ranking"]:
        L.append(f"| {r['rank']} | {r['label']} | {r['sim_cycles']:,} | {'yes' if r['trusted'] else 'no'} |")
    L += ["", f"Search path: {' -> '.join(pred['search']['path'])}. "
          f"Transitions with predictions: {sum(t['sim_verifiable'] for t in pred['transitions'])} simulated, "
          f"{sum(not t['sim_verifiable'] for t in pred['transitions'])} rule-only.", "",
          "## Known simulator limitations (stated before measurement)", ""]
    L += [f"- {x}" for x in pred["known_simulator_limitations"]]
    open(os.path.join(PREREG, "README.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "verify"])
    ap.add_argument("--commit", default="autotune-v1-sim-only")
    a = ap.parse_args()
    os.makedirs(PREREG, exist_ok=True)
    mpath = os.path.join(PREREG, "manifest.json")
    if a.cmd == "write":
        if os.path.exists(mpath):
            sys.exit("manifest.json already exists; a pre-registration is written once. Delete it deliberately to redo.")
        man = manifest(a.commit)
        bad = verify(man)
        if bad:
            sys.exit("working tree differs from the frozen commit:\n  " + "\n  ".join(bad))
        pred = predictions(man)
        json.dump(man, open(mpath, "w"), indent=1)
        json.dump(pred, open(os.path.join(PREREG, "predictions.json"), "w"), indent=1, default=str)
        write_readme(man, pred)
        print(f"wrote {PREREG}: {len(pred['transitions'])} transitions, {len(pred['hypotheses'])} hypotheses")
    else:
        man = json.load(open(mpath))
        bad = verify(man)
        print("frozen files match manifest" if not bad else "MISMATCH:\n  " + "\n  ".join(bad))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
