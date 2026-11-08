"""Predict, then verify.

Before a transformation is simulated, its declared signature (transforms.py) is a prediction of how
the counters will move. After the simulation, `judge` compares prediction and observation counter
by counter and gives one of four verdicts:

  confirmed              counters moved as declared AND the kernel got faster
  mechanism_no_gain      counters moved as declared but cycles did not improve: the transformation
                         did what it says, so the diagnosed bottleneck was not the binding one
  gain_unexplained       faster, but the counters did not move as declared: the rule's model of
                         WHY it helps is wrong (or the simulator is)
  refuted                neither

Beliefs keeps a Beta(successes + 1, failures + 1) confidence per (bottleneck, transformation) pair
and the observed direction of every counter per transformation. The search ranks candidates by
  score = sum over targeted bottlenecks of severity x E[confidence]
so a rule that keeps failing for a bottleneck stops being tried first, and `learned_signature`
reports where a declared signature disagrees with what was observed. All of this is plain data
that can be printed, which is the point: every decision has a reason a person can check.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from .counters import Counters, relative_delta
from .diagnose import Bottleneck
from .space import DesignPoint
from .transforms import Transformation

SIGN_TOL = 0.03      # a counter "moved" if it changed by more than 3%
MIN_GAIN = 0.02      # a step "helped" if cycles dropped by at least 2%
AGREE = 2 / 3        # fraction of declared counters that must match for "moved as declared"


def observed_sign(rel: float) -> str:
    if rel != rel:  # NaN
        return "?"
    if rel > SIGN_TOL:
        return "+"
    if rel < -SIGN_TOL:
        return "-"
    return "0"


@dataclass
class Outcome:
    step: int
    transform: str
    parent: str
    child: str
    targeted: list
    score: float
    predicted: dict
    observed: dict                 # counter -> relative change
    matches: dict                  # counter -> bool (declared, non-'?' counters, excluding cycles)
    agreement: float
    speedup: float                 # parent cycles / child cycles
    verdict: str
    trusted: bool
    flags: list = field(default_factory=list)
    correct: bool = True

    def to_dict(self):
        return asdict(self)


def trust_flags(c: Counters, p: DesignPoint) -> list[str]:
    """Reasons the simulated speed of this point should not be believed without hardware."""
    flags = list(p.sim_blind_spots())
    if c.get("hw_spill_bytes", 0):
        flags.append(f"ptxas spills {c['hw_spill_bytes']} bytes/thread on sm_80; the simulator runs PTX and "
                     "does not charge for spills")
    return flags


def judge(step: int, t: Transformation, parent: DesignPoint, child: DesignPoint, c0: Counters, c1: Counters,
          targeted: list[Bottleneck], score: float) -> Outcome:
    observed, matches = {}, {}
    for k, want in t.expected.items():
        rel = relative_delta(c0, c1, k)
        observed[k] = rel
        if k == "cycles" or want == "?":
            continue
        matches[k] = observed_sign(rel) == want
    agreement = sum(matches.values()) / len(matches) if matches else 1.0
    speedup = c0["cycles"] / c1["cycles"] if c1["cycles"] else 0.0
    helped = speedup >= 1 + MIN_GAIN
    moved = agreement >= AGREE
    verdict = {(True, True): "confirmed", (True, False): "mechanism_no_gain",
               (False, True): "gain_unexplained", (False, False): "refuted"}[(moved, helped)]
    flags = trust_flags(c1, child)
    correct = c1.status == "PASS"
    if not correct:
        verdict, flags = "incorrect", flags + [f"correctness check failed ({c1.status})"]
    return Outcome(step, t.name, parent.key(), child.key(), [b.name for b in targeted], round(score, 3),
                   dict(t.expected), {k: round(v, 4) for k, v in observed.items()}, matches, round(agreement, 3),
                   round(speedup, 4), verdict, trusted=correct and not flags, flags=flags, correct=correct)


class Beliefs:
    def __init__(self):
        self.ab = defaultdict(lambda: [1.0, 1.0])            # (bottleneck, transform) -> [alpha, beta]
        self.signs = defaultdict(lambda: defaultdict(list))  # transform -> counter -> observed signs

    def confidence(self, bottleneck: str, transform: str) -> float:
        a, b = self.ab[(bottleneck, transform)]
        return a / (a + b)

    def score(self, t: Transformation, targeted: list[Bottleneck]) -> float:
        return sum(b.severity * self.confidence(b.name, t.name) for b in targeted)

    def update(self, o: Outcome):
        # Only a trusted, correct result is evidence about speed; flagged results still teach signs.
        for k, rel in o.observed.items():
            self.signs[o.transform][k].append(observed_sign(rel))
        if not o.trusted:
            return
        helped = o.speedup >= 1 + MIN_GAIN
        for b in o.targeted:
            self.ab[(b, o.transform)][0 if helped else 1] += 1

    def learned_signature(self, t: Transformation) -> dict:
        """counter -> (declared, observed signs, verdict) for every counter the transformation declares."""
        out = {}
        for k, want in t.expected.items():
            seen = self.signs[t.name].get(k, [])
            if not seen:
                continue
            maj = max(set(seen), key=seen.count)
            status = "no claim" if want == "?" else ("holds" if all(s == want for s in seen)
                                                      else ("wrong" if maj != want else "mostly holds"))
            out[k] = (want, "".join(seen), status)
        return out

    def to_dict(self):
        return {"confidence": {f"{b}|{t}": v for (b, t), v in self.ab.items()},
                "signs": {t: dict(v) for t, v in self.signs.items()}}

    def save(self, path):
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        b = cls()
        d = json.load(open(path))
        for k, v in d["confidence"].items():
            bn, t = k.split("|")
            b.ab[(bn, t)] = v
        for t, cs in d["signs"].items():
            for k, s in cs.items():
                b.signs[t][k] = list(s)
        return b
