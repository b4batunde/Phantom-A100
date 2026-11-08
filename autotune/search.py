"""Search strategies over the design space, all charged the same way (one simulation per new point).

CounterGuided    diagnose the incumbent -> pick transformations that target its bottlenecks, ranked
                 by severity x learned confidence -> simulate the top `beam` -> judge each against its
                 declared signature -> move to the best trusted improvement -> repeat.
UnguidedLocal    the same transformation menu and acceptance rule, but neighbours are tried in random
                 order with no counters: isolates what the counters contribute.
RandomSearch     uniform sampling of sim-verifiable points without replacement.

Every strategy's answer is the fastest *trusted* point it simulated (correct, and no simulator
blind spot such as register spills). Points the simulator cannot judge at all (double buffering)
are never simulated; the guided search lists them as hardware candidates when its diagnosis
says they target the bottleneck.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .backends import Backend
from .counters import Counters
from .diagnose import diagnose
from .policy import MIN_GAIN, Beliefs, Outcome, judge, trust_flags
from .space import DesignPoint, Workload, all_points
from .transforms import neighbors


@dataclass
class SearchResult:
    strategy: str
    start: str
    evaluated: list = field(default_factory=list)     # (point key, cycles, trusted) in simulation order
    best_curve: list = field(default_factory=list)    # best trusted cycles after k simulations
    path: list = field(default_factory=list)          # incumbents
    outcomes: list = field(default_factory=list)      # Outcome dicts (guided only)
    diagnoses: list = field(default_factory=list)     # per step: incumbent + bottlenecks (guided only)
    hardware_candidates: list = field(default_factory=list)
    stop_reason: str = ""

    @property
    def best(self):
        t = [(cyc, k) for k, cyc, ok in self.evaluated if ok]
        return min(t) if t else (float("inf"), None)


class _Tracker:
    def __init__(self, be: Backend, wl: Workload, res: SearchResult):
        self.be, self.wl, self.res, self.seen = be, wl, res, {}

    def eval(self, p: DesignPoint) -> Counters:
        c = self.be.evaluate(p, self.wl)
        if p.key() not in self.seen:
            self.seen[p.key()] = c
            ok = c.status == "PASS" and not trust_flags(c, p)
            self.res.evaluated.append((p.key(), c["cycles"], ok))
            prev = self.res.best_curve[-1] if self.res.best_curve else float("inf")
            self.res.best_curve.append(min(prev, c["cycles"]) if ok else prev)
        return c

    @property
    def used(self):
        return len(self.seen)


def _blocked(bns, cur: DesignPoint) -> list:
    """Diagnosed bottlenecks for which no transformation in the menu applies to `cur`."""
    applicable = [t for t, _ in neighbors(cur)]
    return [b for b in bns if not any(b.name in t.targets for t in applicable)]


def _unlocks(cur: DesignPoint, child: DesignPoint, blocked) -> list:
    """Transformations that apply to `child` but not to `cur` and target a blocked bottleneck."""
    now = {t.name for t, _ in neighbors(cur)}
    return [(t, b) for t, _ in neighbors(child) if t.name not in now
            for b in blocked if b.name in t.targets]


def counter_guided(start: DesignPoint, wl: Workload, be: Backend, budget: int = 15, beam: int = 2,
                   beliefs: Beliefs | None = None, max_unlock: int = 1) -> tuple[SearchResult, Beliefs]:
    """max_unlock: how many consecutive non-improving 'unlock' moves are allowed. An unlock move is
    taken only when the incumbent has a diagnosed bottleneck that no transformation can address
    from its current structure, and a neighbour makes such a transformation applicable (e.g. the
    tensor-core remedy for cuda_core_math_bound needs a fused kernel). 0 = pure greedy."""
    beliefs = beliefs or Beliefs()
    res = SearchResult("counter_guided", start.key())
    tr = _Tracker(be, wl, res)
    cur, ccur = start, tr.eval(start)
    res.path.append(cur.key())
    step = 0
    unlocks_left = max_unlock
    while True:
        step += 1
        bns = diagnose(ccur, cur)
        res.diagnoses.append({"step": step, "incumbent": cur.key(), "cycles": ccur["cycles"],
                              "bottlenecks": [str(b) for b in bns]})
        cands = []
        for t, child in neighbors(cur):
            targeted = [b for b in bns if b.name in t.targets]
            if not targeted:
                continue
            if not child.sim_verifiable():
                hc = {"from": cur.key(), "transform": t.name, "point": child.key(),
                      "targets": [b.name for b in targeted], "why_not_simulated": child.sim_blind_spots()}
                if hc not in res.hardware_candidates:
                    res.hardware_candidates.append(hc)
                continue
            if child.key() in tr.seen:
                continue
            cands.append((beliefs.score(t, targeted), t, child, targeted))
        cands.sort(key=lambda x: (-x[0], x[1].name))
        improved: list[tuple[Outcome, DesignPoint, Counters]] = []
        i = 0
        while i < len(cands) and tr.used < budget:
            for score, t, child, targeted in cands[i:i + beam]:
                if tr.used >= budget:
                    break
                c = tr.eval(child)
                o = judge(step, t, cur, child, ccur, c, targeted, score)
                beliefs.update(o)
                res.outcomes.append(o.to_dict())
                if o.trusted and o.speedup >= 1 + MIN_GAIN:
                    improved.append((o, child, c))
            i += beam
            if improved:
                break
        if not improved and tr.used < budget and unlocks_left > 0:
            blocked = _blocked(bns, cur)
            opts = []
            for t, child in neighbors(cur):
                if not child.sim_verifiable():
                    continue
                u = _unlocks(cur, child, blocked)
                if u:
                    opts.append((sum(b.severity * beliefs.confidence(b.name, ut.name) for ut, b in u), t, child, u))
            opts.sort(key=lambda x: (-x[0], x[1].name))
            if opts:
                _, t, child, u = opts[0]
                c = tr.eval(child)
                if c.status == "PASS":
                    unlocks_left -= 1
                    res.diagnoses[-1]["unlock"] = (
                        f"no trusted improvement; blocked bottleneck(s) {', '.join(sorted({b.name for _, b in u}))} "
                        f"have no remedy from {cur.key()}. Taking non-improving `{t.name}` -> {child.key()} "
                        f"({ccur['cycles'] / c['cycles']:.2f}x) because it makes "
                        f"{', '.join(sorted({ut.name for ut, _ in u}))} applicable")
                    cur, ccur = child, c
                    res.path.append(cur.key())
                    continue
        if not improved:
            if tr.used >= budget:
                res.stop_reason = f"step {step}: budget exhausted"
            elif not cands:
                res.stop_reason = (f"step {step}: no untried transformation in the menu targets the diagnosed "
                                   f"bottlenecks of {cur.key()}")
            else:
                res.stop_reason = (f"step {step}: none of the {len(cands)} targeted transformations of {cur.key()} "
                                   f"gave a trusted improvement of >= {100 * MIN_GAIN:.0f}%")
            break
        o, cur, ccur = max(improved, key=lambda x: x[0].speedup)
        unlocks_left = max_unlock
        res.path.append(cur.key())
        if tr.used >= budget:
            res.stop_reason = f"step {step}: budget exhausted"
            break
    return res, beliefs


def unguided_local(start: DesignPoint, wl: Workload, be: Backend, budget: int = 15,
                   seed: int = 0) -> SearchResult:
    rng = random.Random(seed)
    res = SearchResult("unguided_local", start.key())
    tr = _Tracker(be, wl, res)
    cur, ccur = start, tr.eval(start)
    res.path.append(cur.key())
    while tr.used < budget:
        nb = [q for _, q in neighbors(cur) if q.sim_verifiable() and q.key() not in tr.seen]
        rng.shuffle(nb)
        moved = False
        for q in nb:
            if tr.used >= budget:
                break
            c = tr.eval(q)
            if c.status == "PASS" and not trust_flags(c, q) and ccur["cycles"] / c["cycles"] >= 1 + MIN_GAIN:
                cur, ccur, moved = q, c, True
                res.path.append(cur.key())
                break
        if not moved:
            res.stop_reason = "local optimum" if tr.used < budget else "budget exhausted"
            break
    return res


def random_search(start: DesignPoint, wl: Workload, be: Backend, budget: int = 15, seed: int = 0) -> SearchResult:
    rng = random.Random(seed)
    res = SearchResult("random", start.key())
    tr = _Tracker(be, wl, res)
    tr.eval(start)
    pool = [p for p in all_points(include_blind=False) if p != start]
    rng.shuffle(pool)
    for p in pool:
        if tr.used >= budget:
            break
        tr.eval(p)
    res.stop_reason = "budget exhausted" if tr.used >= budget else "space exhausted"
    return res
