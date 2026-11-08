"""Coverage / conditional accuracy decomposition of direction predictions (ADDENDUM-2).

    python -m autotune.decompose            # reads results/autotune/measured/scores.json

One "direction accuracy" number mixes two different failure modes: a predictor that says "no change"
where hardware shows a difference (insensitive), and a predictor that commits to the wrong direction
(confidently wrong). This module reports them separately, for the simulator and for the declared
signatures, per counter, per mapping equivalence and per transformation family.

For one predictor and one group of scored transitions (hardware observable, predictor made a claim):

  coverage              committed predictions (up/down, faster/slower) / all claims (incl. tie)
  conditional accuracy  committed predictions in the hardware direction / committed predictions
  confidently wrong     committed predictions opposite to hardware / committed predictions
                        (the rest of the committed misses are hardware ties)
  tie resolution        predicted ties that hardware resolved into a real difference / predicted ties
  overall accuracy      matches / all claims (the pre-registered metric, unchanged)

Tolerances are the pre-registered ones (3% counters, 2% performance). Writes decomposition.{json,md}
next to scores.json and never modifies scores.json or validation.md.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from .backends import REPO
from .hw_counters import MAPPING, UNOBSERVABLE

MEASURED = os.path.join(REPO, "results", "autotune", "measured")

# Fixed in ADDENDUM-2, before the Nsight run.
FAMILIES = {
    "structural": ("tile_smem", "fuse", "vectorize_fp16", "tensor_cores"),
    "tuning": ("reg_cap_128", "remove_reg_cap", "grow_br", "shrink_br", "grow_bc", "shrink_bc"),
    "rule_only": ("double_buffer",),
}
FAMILY_OF = {t: f for f, ts in FAMILIES.items() for t in ts}
PERF = {"faster": "+", "slower": "-", "same": "0"}


def norm(x):
    """Map a prediction or observation to '+', '-', '0', or None (no claim / unobservable)."""
    if x is None or x == UNOBSERVABLE or x == "?":
        return None
    return PERF.get(x, x)


def tally(pairs):
    """pairs: (predicted, observed) already normalized, both not None."""
    c = defaultdict(int)
    for p, o in pairs:
        c["claims"] += 1
        if p == "0":
            c["ties"] += 1
            c["tie_resolved"] += o != "0"
        else:
            c["committed"] += 1
            c["committed_right"] += p == o
            c["confidently_wrong"] += o not in ("0", p)
        c["match"] += p == o

    def r(a, b):
        return c[a] / c[b] if c[b] else None

    return {"claims": c["claims"], "committed": c["committed"], "ties": c["ties"],
            "coverage": r("committed", "claims"), "conditional_accuracy": r("committed_right", "committed"),
            "confidently_wrong": r("confidently_wrong", "committed"), "tie_resolution": r("tie_resolved", "ties"),
            "overall_accuracy": r("match", "claims"),
            "counts": {k: c[k] for k in ("committed_right", "confidently_wrong", "tie_resolved", "match")}}


def decompose(scores: dict) -> dict:
    rows = []  # (predictor, counter, equivalence, family, predicted, observed)
    for t in scores["transitions"]:
        fam = FAMILY_OF.get(t["transform"], "other")
        for k, r in t["rows"].items():
            o = norm(r["hw"])
            if o is None:
                continue
            for pred_name, key in (("simulator", "sim_pred"), ("declared", "declared")):
                if pred_name == "simulator" and not t["sim_verifiable"]:
                    continue
                p = norm(r.get(key))
                if p is not None:
                    rows.append((pred_name, k, r["equivalence"], fam, p, o))
    out = {}
    for pred_name in ("simulator", "declared"):
        R = [x for x in rows if x[0] == pred_name]
        out[pred_name] = {
            "by_counter": {k: tally([(p, o) for _, kk, _, _, p, o in R if kk == k])
                           for k in [*[m for m in MAPPING if m != "cycles"], "performance"]},
            "by_equivalence": {e: tally([(p, o) for _, _, ee, _, p, o in R if ee == e])
                               for e in ("same_definition", "approximate", "proxy", "performance")},
            "by_family": {f: tally([(p, o) for _, _, _, ff, p, o in R if ff == f]) for f in FAMILIES},
            "by_family_and_counter": {f: {k: tally([(p, o) for _, kk, _, ff, p, o in R if ff == f and kk == k])
                                          for k in [*[m for m in MAPPING if m != "cycles"], "performance"]}
                                      for f in FAMILIES},
        }
    return out


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def table(title, d, label=lambda k: k):
    L = [f"#### {title}", "",
         "| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |",
         "|---|---|---|---|---|---|---|"]
    for k, s in d.items():
        if not s["claims"]:
            continue
        L.append(f"| {label(k)} | {s['claims']} | {pct(s['coverage'])} ({s['committed']}) | "
                 f"{pct(s['conditional_accuracy'])} ({s['counts']['committed_right']}/{s['committed']}) | "
                 f"{pct(s['confidently_wrong'])} | {pct(s['tie_resolution'])} ({s['counts']['tie_resolved']}/{s['ties']}) | "
                 f"{pct(s['overall_accuracy'])} |")
    return L + [""]


def report(dec: dict, meas_note: str) -> str:
    lab = lambda k: MAPPING[k]["label"] if k in MAPPING else "Performance direction"
    L = ["# Coverage and conditional accuracy (Measured; decomposition defined in ADDENDUM-2)", "", meas_note, "",
         "Coverage = share of claims where the predictor committed to a direction instead of a tie. "
         "Conditional accuracy = share of committed predictions in the hardware direction. Confidently wrong = "
         "committed predictions opposite to hardware. Tie resolution = predicted ties that hardware resolved "
         "into a real difference. Overall accuracy = the pre-registered metric. Unobservable hardware values "
         "are excluded. Counts in parentheses.", ""]
    for pred in ("simulator", "declared"):
        name = "Simulator (observed simulated direction)" if pred == "simulator" else "Declared signatures (rules)"
        L += [f"## {name}", ""]
        L += table("By transformation family", dec[pred]["by_family"])
        L += table("By counter", dec[pred]["by_counter"], lab)
        L += table("By mapping equivalence", dec[pred]["by_equivalence"])
        for f, d in dec[pred]["by_family_and_counter"].items():
            if any(s["claims"] for s in d.values()):
                L += table(f"Family `{f}` by counter", d, lab)
    return "\n".join(L) + "\n"


def main():
    path = os.path.join(MEASURED, "scores.json")
    if not os.path.exists(path):
        sys.exit(f"no {path}; run python -m autotune.validate_hw first")
    scores = json.load(open(path))
    meas = os.path.join(MEASURED, "hw_measurements.json")
    note = ""
    if os.path.exists(meas):
        m = json.load(open(meas))
        note = (f"Source: `scores.json` from {m['gpu']}, BH={m['workload']['BH']}, N={m['workload']['N']}; "
                f"Nsight Compute: {m['profiling']}.")
    dec = decompose(scores)
    json.dump(dec, open(os.path.join(MEASURED, "decomposition.json"), "w"), indent=1)
    rep = report(dec, note)
    open(os.path.join(MEASURED, "decomposition.md"), "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
