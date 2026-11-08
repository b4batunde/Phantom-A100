"""CPU-only tests for autotune/ (no GPU, no simulator). Synthetic counters below are test fixtures,
not results."""
import glob
import os

import pytest

from autotune.backends import Backend
from autotune.counters import Counters, parse_log
from autotune.diagnose import diagnose
from autotune.policy import Beliefs, judge
from autotune.search import counter_guided, random_search, unguided_local
from autotune.space import HISTORICAL, DesignPoint, Workload, all_points, make
from autotune.transforms import BY_NAME, MENU, neighbors
from autotune.validate_hw import kendall, spearman

WL = Workload()


def test_space_size_and_keys():
    pts = all_points()
    assert len(pts) == 23 and len(all_points(include_blind=False)) == 15
    assert len({p.key() for p in pts}) == len(pts)
    assert len({p.harness_spec() for p in pts}) == len(pts)
    for p in pts:
        assert DesignPoint.from_key(p.key()) == p


def test_historical_versions_are_in_the_space():
    for name, kw in HISTORICAL.items():
        p = make(**kw)
        assert p in all_points(), name
        assert p.historical_name() == name
    assert not make("fused_vec", db=1).sim_verifiable()
    # the original kernels keep their original harness names
    assert make("fused").harness_spec() == "fused"
    assert make("fused_vec").harness_spec() == "fused_tuned:64:32:0:0"


def test_transformations_stay_in_space():
    for p in all_points():
        for t, q in neighbors(p):
            assert q.is_valid() and q != p and q in all_points(), (t.name, p, q)
    assert {t.name for t, _ in neighbors(make("naive"))} == {"tile_smem", "fuse"}
    for t in MENU:
        assert set(t.expected.values()) <= {"+", "-", "0", "?"}


def C(**kw):
    raw = dict(cycles=1000, thread_insn=1000, warp_insn=100, occupancy=20.0, regs=200, l2_accesses=100,
               shmem_insn=10, global_load_insn=10, shmem_bank_conflicts=0, kernel_launches=1, cta_per_sm=4,
               cta_limiter="regs", hw_spill_bytes=0)
    der = dict(l2_traffic_ratio=1.5, insn_per_flop=0.3, scoreboard_frac=0.3, smem_conflict_rate=0.0,
               waves=2.0, dram_sectors=0, blocks=84)
    for k, v in kw.items():
        (der if k in der else raw)[k] = v
    return Counters("PASS", 0.0, raw, der)


def test_diagnose_rules():
    names = [b.name for b in diagnose(C(occupancy=10.0), make("fused"))]
    assert "register_limited_occupancy" in names
    names = [b.name for b in diagnose(C(kernel_launches=3, l2_traffic_ratio=9.0), make("naive"))]
    assert {"materialized_intermediate", "redundant_global_traffic"} <= set(names)
    assert diagnose(C(occupancy=40.0), make("fused_tc")) == []


def test_judge_verdicts():
    t = BY_NAME["vectorize_fp16"]
    p0, p1 = make("fused"), make("fused_vec")
    c0 = C(occupancy=10.0)
    tg = diagnose(c0, p0)
    ok = judge(1, t, p0, p1, c0, C(cycles=500, global_load_insn=2, l2_accesses=30, regs=150, occupancy=20.0), tg, 1)
    assert ok.verdict == "confirmed" and ok.trusted
    nogain = judge(1, t, p0, p1, c0, C(cycles=1000, global_load_insn=2, l2_accesses=30, regs=150, occupancy=20.0), tg, 1)
    assert nogain.verdict == "mechanism_no_gain"
    lucky = judge(1, t, p0, p1, c0, C(cycles=500), tg, 1)
    assert lucky.verdict == "gain_unexplained"
    spill = judge(1, BY_NAME["reg_cap_128"], p0, make("fused", maxreg=128), c0,
                  C(cycles=400, regs=128, occupancy=30.0, cta_per_sm=6, hw_spill_bytes=188), tg, 1)
    assert not spill.trusted and spill.flags


def test_beliefs_learn_from_failures():
    b = Beliefs()
    t = BY_NAME["reg_cap_128"]
    p0, p1 = make("fused_vec"), make("fused_vec", maxreg=128)
    c0 = C(occupancy=10.0)
    tg = diagnose(c0, p0)
    before = b.score(t, tg)
    b.update(judge(1, t, p0, p1, c0, C(cycles=1100, regs=128, occupancy=20.0, cta_per_sm=6), tg, 1))
    assert b.score(t, tg) < before
    sig = b.learned_signature(t)
    assert sig["regs"][2] == "holds"


class FakeBackend(Backend):
    """Synthetic landscape: tensor cores best, register cap looks great but spills."""

    def __init__(self):
        super().__init__()
        base = {
            "naive": C(cycles=100_000, kernel_launches=3, l2_traffic_ratio=18.0, insn_per_flop=3.0, occupancy=47.0,
                       cta_limiter="regs"),
            "tiled": C(cycles=30_000, kernel_launches=3, l2_traffic_ratio=8.0, insn_per_flop=2.5, occupancy=50.0),
            "fused_tc": C(cycles=2_000, insn_per_flop=0.13, occupancy=37.0, regs=80, cta_limiter="smem"),
        }
        self.table = {}
        for p in all_points(include_blind=False):
            if p.key() in base:
                self.table[p.key()] = base[p.key()]
            elif p.maxreg:
                self.table[p.key()] = C(cycles=10_000, insn_per_flop=1.5, regs=128, occupancy=24.0, hw_spill_bytes=168)
            elif p.structure == "fused":
                self.table[p.key()] = C(cycles=40_000 + p.br, insn_per_flop=1.3, l2_traffic_ratio=7.8,
                                        occupancy=9.8, regs=240, kernel_launches=1, l2_accesses=700)
            else:
                self.table[p.key()] = C(cycles=20_000 + p.br + p.bc, insn_per_flop=2.0, occupancy=18.7, regs=168,
                                        global_load_insn=2, l2_accesses=170)

    def evaluate(self, p, wl):
        if p.key() not in self.calls:
            self.calls.append(p.key())
        return self.table[p.key()]


def test_guided_search_on_synthetic_landscape():
    be = FakeBackend()
    res, beliefs = counter_guided(make("naive"), WL, be, budget=15)
    assert res.path[-1] == "fused_tc"
    assert "fused/br64/bc32/r128" not in res.path  # spilling point never becomes the incumbent
    assert res.best[1] == "fused_tc"
    assert any("unlock" in d for d in res.diagnoses)
    for o in res.outcomes:
        assert o["verdict"] in {"confirmed", "mechanism_no_gain", "gain_unexplained", "refuted", "incorrect"}


def test_pure_greedy_stops_at_the_fusion_barrier():
    # Fusion is slower than tiling in this landscape (as at N=128 in simulation), so a greedy
    # search with no unlock move never reaches the tensor-core kernel.
    res, _ = counter_guided(make("naive"), WL, FakeBackend(), budget=15, max_unlock=0)
    assert res.path == ["naive", "tiled"]


def test_baselines_respect_budget():
    for fn in (random_search, unguided_local):
        be = FakeBackend()
        r = fn(make("naive"), WL, be, budget=5, seed=3)
        assert len(r.evaluated) <= 5 and len(r.best_curve) == len(r.evaluated)


def test_rank_correlations():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


LOG_TAR = os.path.join(os.path.dirname(__file__), "..", "results", "autotune", "simulator_logs.tar.gz")


@pytest.mark.skipif(not os.path.exists(LOG_TAR), reason="no simulator logs in results/autotune/")
def test_parse_real_simulator_logs():
    import tarfile
    by_file = {q.key().replace("/", "_"): q for q in all_points()}
    n = 0
    with tarfile.open(LOG_TAR) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".log"):
                continue
            key = os.path.basename(m.name).split("__", 1)[1][:-4]
            p = by_file[key]
            c = parse_log(tf.extractfile(m).read().decode(), p, WL)
            assert c.status == "PASS" and c["cycles"] > 0 and c["regs"] > 0 and c["cta_per_sm"] > 0, m.name
            n += 1
    assert n == 15
