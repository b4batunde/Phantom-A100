"""CPU-only tests for the pre-registration and hardware-validation scoring.

The "hardware" measurements below are SYNTHETIC FIXTURES built from simulated counters with
deliberate perturbations. They exist only to exercise the scoring code; they are not results."""
import copy
import json
import os

import pytest

from autotune.backends import REPO
from autotune.freeze import PREREG, verify
from autotune.hw_counters import MAPPING, UNOBSERVABLE, aggregate, parse_ncu_csv
from autotune.space import all_points
from autotune.validate_hw import hw_dir, match, report, score

PRED = os.path.join(PREREG, "predictions.json")
pytestmark = pytest.mark.skipif(not os.path.exists(PRED), reason="no pre-registration")


def load():
    man = json.load(open(os.path.join(PREREG, "manifest.json")))
    pred = json.load(open(PRED))
    cache = {v["point"]: v["counters"] for v in
             json.load(open(os.path.join(REPO, "results", "autotune", "sim_cache.json"))).values()}
    static = json.load(open(os.path.join(REPO, "results", "autotune", "static_sm80.json")))
    return man, pred, cache, static


def test_frozen_files_match_manifest():
    man, *_ = load()
    assert verify(man) == []
    assert man["hardware_measurements_observed"] is False


def test_predictions_are_complete():
    _, pred, _, _ = load()
    assert len(pred["simulated_ranking"]) == 15
    assert pred["search"]["path"][0] == "naive" and pred["search"]["path"][-1] == "fused_tc"
    assert {h["id"] for h in pred["hypotheses"]} == {f"H{i}" for i in range(1, 10)}
    h3 = next(h for h in pred["hypotheses"] if h["id"] == "H3")
    assert "242" in h3["risk_identified_before_measurement"] and "168" in h3["risk_identified_before_measurement"]
    assert pred["untrusted_candidates"] and all(not r["trusted"] for r in pred["untrusted_candidates"])
    assert set(pred["counter_mapping"]) == set(MAPPING)
    for t in pred["transitions"]:
        if t["sim_verifiable"]:
            assert set(t["sim_counter_direction"]) == set(MAPPING) - {"cycles"}


def fake_hw(cache, static, with_ncu=True, flip_v2v3=False, scoreboard=False):
    """SYNTHETIC hardware measurements derived from simulated counters (test fixture only)."""
    pts = {}
    for p in all_points():
        k = p.key()
        base = p.key().replace("/db", "")
        c = cache.get(k) or cache.get(base)
        ms = c["cycles"] / 1e6 * (0.93 if "/db" in k else 1.0)
        if flip_v2v3 and k == "fused/br64/bc32":
            ms = cache["tiled"]["cycles"] / 1e6 * 0.8
        ncu = None
        if with_ncu:
            s = 324 / 42  # counts scale with the larger hardware workload
            ncu = {"regs": c["regs"], "occupancy": c["occupancy"], "cta_per_sm": c["cta_per_sm"],
                   "blocks": (c.get("blocks") or 672) * s, "kernel_launches": c["kernel_launches"],
                   "l2_accesses": c["l2_accesses"] * s, "dram_sectors": c["dram_sectors"] * s,
                   "global_load_insn": c["global_load_insn"] * s, "shmem_insn": c["shmem_insn"] * s,
                   "warp_insn": c["warp_insn"] * s, "thread_insn": c["thread_insn"] * s,
                   "shmem_bank_conflicts": c["shmem_bank_conflicts"] * s, "issue_util": 100 * c["issue_util"],
                   "scoreboard_frac": 100 * c["scoreboard_frac"] if scoreboard else None,
                   "cycles": None, "cta_limiter": c["cta_limiter"]}
        pts[k] = {"status": "PASS", "median_ms": ms, "min_ms": ms, "ncu": ncu, "dropped_metrics": []}
    return {"gpu": "SYNTHETIC FIXTURE", "sms": 108, "arch": "80", "workload": {"BH": 324, "N": 128, "causal": 0},
            "bh_scale": 108 / 14, "iters": 1, "profiling": "ok" if with_ncu else "no_permission",
            "points": pts, "static_table": static}


def test_scoring_with_counters():
    man, pred, cache, static = load()
    sc = score(fake_hw(cache, static), pred, cache, [])
    h = sc["hypotheses"]
    assert h["H1"]["result"] == "supported"          # fixture time is proportional to simulated cycles
    assert h["H3"]["result"] == "simulator ordering holds"
    assert h["H8"]["result"] == "supported"          # fixture makes db 7% faster
    pc = sc["mechanism_fidelity"]["per_counter"]
    assert pc["regs"]["simulator"]["accuracy"] == 1.0
    assert pc["scoreboard_frac"]["simulator"]["match"] == 0
    assert pc["scoreboard_frac"]["simulator"]["unobservable"] > 0   # missing -> UNOBSERVABLE, not mismatch
    assert pc["scoreboard_frac"]["simulator"]["mismatch"] == 0
    assert "Simulator prediction accuracy" in report(fake_hw(cache, static), pred, sc)


def test_perfect_fidelity_fixture_reproduces_the_simulated_search():
    # If hardware behaved exactly like the simulator, the frozen search rerun on hardware data
    # must take the simulated path. Guards against unit/scale bugs in the hardware adapter.
    man, pred, cache, static = load()
    sc = score(fake_hw(cache, static, scoreboard=True), pred, cache, [])
    hw = sc["decision_fidelity"]["search_on_hardware"]
    assert hw["path"] == pred["search"]["path"]
    assert sc["hypotheses"]["H9"]["result"] == "supported"
    assert sc["decision_fidelity"]["greedy_on_hardware"]["path"] == pred["search"]["greedy_ablation_path"]
    agree = [c["jaccard"] for c in sc["decision_fidelity"]["classification"]]
    assert min(agree) == 1.0


def test_scoring_timing_only_and_reversal():
    man, pred, cache, static = load()
    meas = fake_hw(cache, static, with_ncu=False, flip_v2v3=True)
    sc = score(meas, pred, cache, [])
    h = sc["hypotheses"]
    assert h["H3"]["result"] == "reversed"
    assert h["H9"]["result"].startswith("inconclusive")
    assert h["H7"]["result"].startswith("inconclusive")
    occ = sc["mechanism_fidelity"]["per_counter"]["occupancy"]["simulator"]
    assert occ["accuracy"] is None and occ["unobservable"] > 0
    # registers stay observable through the static sm_80 build
    assert sc["mechanism_fidelity"]["per_counter"]["regs"]["simulator"]["match"] > 0
    report(meas, pred, sc)


def test_direction_and_match_helpers():
    assert hw_dir(None, 3) == UNOBSERVABLE
    assert hw_dir(100, 102) == "0" and hw_dir(100, 110) == "+" and hw_dir(0, 5) == "+"
    assert match("?", "+") == "NO_CLAIM" and match("+", UNOBSERVABLE) == UNOBSERVABLE
    assert match("-", "-") == "MATCH" and match("-", "+") == "MISMATCH"


NCU_CSV = '''==PROF== Connected to process 1
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","gpu__time_duration.sum","launch__registers_per_thread","lts__t_sectors.sum"
"","","","","","","","nsecond","register/thread","sector"
"0","1","harness","h","scores_tiled","1","7","1,000","32","5000"
"1","1","harness","h","softmax_warp","1","7","200","20","800"
"2","1","harness","h","pv_tiled","1","7","800","32","4000"
'''


def test_parse_and_aggregate_ncu_csv():
    ks = parse_ncu_csv(NCU_CSV)  # synthetic CSV in ncu's --page raw layout
    assert len(ks) == 3 and ks[0]["gpu__time_duration.sum"] == 1000.0
    agg = aggregate(ks)
    assert agg["l2_accesses"] == 9800 and agg["regs"] == 32 and agg["kernel_launches"] == 3
    assert agg["occupancy"] is None and agg["cta_limiter"] is None   # not collected -> unobservable
