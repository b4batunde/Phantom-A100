"""Experiments on the simulated ground truth (all numbers Simulated, GPGPU-Sim A100 slice).

    python -m autotune.experiments [--BH 42 --N 128 --causal 0] [--seeds 500]

1. Rediscovery: counter-guided search from v1 naive, never told the v1 -> v4 path. Full decision
   trace (diagnosis, ranked candidates, predicted vs observed counters, verdicts).
2. Robustness: the same search from every other sim-verifiable starting point.
3. Same-budget comparison: counter-guided vs unguided local search vs random search, from v1, at
   every simulation budget (random strategies averaged over --seeds seeds).
4. Signature audit: which declared counter signatures held, and which were wrong.

Writes results/autotune/{README.md, experiments.json, budget_curve.png}. Runs entirely from the
simulation cache (ReplayBackend): run autotune.groundtruth first.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from .backends import REPO, ReplayBackend
from .diagnose import THRESHOLDS
from .policy import AGREE, MIN_GAIN, SIGN_TOL, Beliefs
from .search import counter_guided, random_search, unguided_local
from .space import HISTORICAL, DesignPoint, Workload, all_points, make
from .transforms import BY_NAME, MENU

OUT = os.path.join(REPO, "results", "autotune")


def fmt_pt(k: str) -> str:
    return DesignPoint.from_key(k).label()


def sims_to_reach(curve: list, target: float) -> int | None:
    for i, v in enumerate(curve):
        if v <= target:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--BH", type=int, default=42)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--beam", type=int, default=2)
    a = ap.parse_args()
    wl = Workload(a.BH, a.N, a.causal)
    be = ReplayBackend()
    pts = all_points(include_blind=False)
    missing = [p.key() for p in pts if not be.has(p, wl)]
    assert not missing, f"not simulated yet: {missing} (run python -m autotune.groundtruth)"
    budget = len(pts)

    # ---------------------------------------------------------------- ground truth
    from .policy import trust_flags
    gt = []
    for p in pts:
        c = be.evaluate(p, wl)
        gt.append(dict(point=p.key(), label=p.label(), cycles=c["cycles"], status=c.status,
                       occupancy=round(c["occupancy"], 1), regs=c["regs"], hw_regs=c.get("hw_regs", 0),
                       hw_spill=c.get("hw_spill_bytes", 0), cta=c["cta_per_sm"], limiter=c["cta_limiter"],
                       l2x=round(c["l2_traffic_ratio"], 2), ipf=round(c["insn_per_flop"], 2),
                       trusted=c.status == "PASS" and not trust_flags(c, p)))
    gt.sort(key=lambda r: r["cycles"])
    trusted = [r for r in gt if r["trusted"]]
    opt = trusted[0]
    target = opt["cycles"] * 1.05
    naive_cyc = next(r["cycles"] for r in gt if r["point"] == "naive")

    # ---------------------------------------------------------------- 1. rediscovery
    be.reset()
    red, beliefs = counter_guided(make("naive"), wl, be, budget=budget, beam=a.beam)
    be.reset()
    greedy, _ = counter_guided(make("naive"), wl, be, budget=budget, beam=a.beam, max_unlock=0)

    # ---------------------------------------------------------------- 2. robustness
    robust = []
    for p in pts:
        be.reset()
        r, _ = counter_guided(p, wl, be, budget=budget, beam=a.beam)
        bc, bk = r.best
        robust.append(dict(start=p.key(), sims=len(r.evaluated), best=bk, best_cycles=bc,
                           within_5pct=bc <= target, path=r.path, stop=r.stop_reason))

    # ---------------------------------------------------------------- 3. same budget
    def curve_pad(c):
        return c + [c[-1]] * (budget - len(c))

    g_curve = curve_pad(red.best_curve)
    strat = {"counter_guided": [g_curve]}
    reach = {"counter_guided": [sims_to_reach(red.best_curve, target)]}
    for name, fn in (("unguided_local", unguided_local), ("random", random_search)):
        strat[name], reach[name] = [], []
        for s in range(a.seeds):
            be.reset()
            r = fn(make("naive"), wl, be, budget=budget, seed=s)
            strat[name].append(curve_pad(r.best_curve))
            reach[name].append(sims_to_reach(r.best_curve, target))

    def summarize(name):
        rs = reach[name]
        hit = [x for x in rs if x is not None]
        return dict(p_reach=len(hit) / len(rs), mean_sims=statistics.mean(hit) if hit else None,
                    median_sims=statistics.median(hit) if hit else None,
                    final_mean_cycles=statistics.mean(c[-1] for c in strat[name]),
                    curve_mean=[statistics.mean(c[k] for c in strat[name]) for k in range(budget)],
                    speedup_curve_mean=[statistics.mean(naive_cyc / c[k] for c in strat[name]) for k in range(budget)])

    summary = {k: summarize(k) for k in strat}

    # ---------------------------------------------------------------- 4. signature audit
    audit = {t.name: beliefs.learned_signature(t) for t in MENU}

    os.makedirs(OUT, exist_ok=True)
    data = dict(workload=wl.key(), budget=budget, ground_truth=gt, optimum=opt, target_cycles=target,
                rediscovery=red.__dict__, ablation_no_unlock=greedy.__dict__, robustness=robust, budget_comparison=summary,
                signature_audit={k: {c: list(v) for c, v in d.items()} for k, d in audit.items()},
                beliefs=beliefs.to_dict(), seeds=a.seeds, beam=a.beam)
    json.dump(data, open(os.path.join(OUT, "experiments.json"), "w"), indent=1, default=str)
    beliefs.save(os.path.join(OUT, "beliefs.json"))
    plot(summary, budget, naive_cyc, opt["cycles"], os.path.join(OUT, "budget_curve.png"), wl, a.seeds)
    write_readme(data, wl, naive_cyc, a)
    print(open(os.path.join(OUT, "README.md")).read())


def plot(summary, budget, naive_cyc, opt_cyc, path, wl, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, quiet, grey = "#1f1f1f", "#6b6b6b", "#b8b8b8"
    colors = {"counter_guided": "#2a6fdb", "unguided_local": "#e8871e", "random": "#8c8c8c"}
    names = {"counter_guided": "counter-guided (deterministic)",
             "unguided_local": f"unguided local search (mean of {seeds} seeds)",
             "random": f"random search (mean of {seeds} seeds)"}
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    xs = list(range(1, budget + 1))
    for k in ("random", "unguided_local", "counter_guided"):
        ys = summary[k]["speedup_curve_mean"]
        ax.plot(xs, ys, color=colors[k], lw=2.4 if k == "counter_guided" else 1.6, label=names[k],
                marker="o" if k == "counter_guided" else None, ms=4)
    ax.axhline(naive_cyc / opt_cyc, color=grey, lw=1, ls="--")
    ax.text(budget, naive_cyc / opt_cyc, " best trusted point", va="center", fontsize=8, color=quiet)
    ax.set_xlabel("simulations used (each one ~1-45 min of CPU time)", color=quiet)
    ax.set_ylabel("best trusted speedup over v1 found so far", color=quiet)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Simulated: speedup found vs. simulation budget, starting from v1", loc="left", fontsize=11,
                 fontweight="bold", color=ink)
    fig.text(0.01, 0.01, f"GPGPU-Sim A100 model (14-SM slice), {wl.key()}, d=64, FP16. "
             "Trusted = correct and no simulator blind spot (e.g. register spills).", fontsize=7, color=quiet)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def write_readme(d, wl, naive_cyc, a):
    L = []
    w = L.append
    gt, opt = d["ground_truth"], d["optimum"]
    red = d["rediscovery"]
    w("# Counter-guided optimization: results (Simulated)")
    w("")
    w(f"Every number here is **Simulated**: GPGPU-Sim (patched, `sim/patches/`) on its SM80_A100 model cut to a "
      f"14-SM slice, workload {wl.key()} (BH = batch x heads, N = sequence length, d = 64, FP16). "
      "No real GPU was used. Generated by `python -m autotune.experiments`.")
    w("")
    w("## Ground truth: every sim-verifiable design point")
    w("")
    w(f"The design space has {len(all_points())} points; {len(gt)} can be judged by the simulator (the "
      "double-buffered ones cannot: no `cp.async`). All of them were simulated once to score the searches. "
      "*Trusted* = passes the correctness check and has no known simulator blind spot.")
    w("")
    w("| Rank | Design point | Cycles | Speedup vs v1 | Occupancy | Regs (sim / sm_80) | Spills (sm_80) | Blocks/SM (limit) | L2 traffic x | Insn/FLOP | Trusted |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(gt, 1):
        w(f"| {i} | {r['label']} | {r['cycles']:,} | {naive_cyc / r['cycles']:.1f}x | {r['occupancy']}% | "
          f"{r['regs']} / {r['hw_regs']} | {r['hw_spill']} B | {r['cta']} ({r['limiter']}) | {r['l2x']} | "
          f"{r['ipf']} | {'yes' if r['trusted'] else '**no**'} |")
    w("")
    w(f"Best trusted point: **{opt['label']}**, {opt['cycles']:,} cycles "
      f"({naive_cyc / opt['cycles']:.1f}x over v1). Target for the comparisons below: within 5% of it.")
    w("")
    w("## 1. Rediscovery from v1 (never told the v1 -> v4 path)")
    w("")
    hist = [k for k in red["path"] if DesignPoint.from_key(k).historical_name()]
    w(f"Path taken: {' -> '.join(fmt_pt(k) for k in red['path'])}")
    w("")
    w(f"Simulations used: {len(red['evaluated'])} of {len(gt)}. Stop reason: {red['stop_reason']}.")
    w("")
    w(f"Historical versions on the path: {', '.join(DesignPoint.from_key(k).historical_name() for k in hist)}.")
    w("")
    g = d["ablation_no_unlock"]
    gb = g["evaluated"] and min((c, k) for k, c, ok in g["evaluated"] if ok)
    w(f"Ablation, pure greedy (no unlock move): {' -> '.join(fmt_pt(k) for k in g['path'])}; "
      f"best trusted {fmt_pt(gb[1])} at {gb[0]:,} cycles after {len(g['evaluated'])} simulations. "
      f"Stop reason: {g['stop_reason']}.")
    w("")
    w("### Decision trace")
    w("")
    for dg in red["diagnoses"]:
        w(f"**Step {dg['step']}**: incumbent `{dg['incumbent']}` ({dg['cycles']:,} cycles). Diagnosed:")
        w("")
        for b in dg["bottlenecks"] or ["(nothing past its threshold)"]:
            w(f"- {b}")
        w("")
        outs = [o for o in red["outcomes"] if o["step"] == dg["step"]]
        unlock = dg.get("unlock")
        if outs:
            w("| Tried | Targets | Score | Predicted -> observed (counter: declared / change) | Agreement | Speedup | Verdict | Trusted |")
            w("|---|---|---|---|---|---|---|---|")
            for o in outs:
                pv = "; ".join(f"{k}: {v} / {o['observed'][k] * 100:+.0f}%" if abs(o['observed'][k]) != float('inf')
                               else f"{k}: {v} / new" for k, v in o["predicted"].items())
                tr = "yes" if o["trusted"] else "no: " + "; ".join(o["flags"])
                w(f"| `{o['transform']}` -> `{o['child']}` | {', '.join(o['targeted'])} | {o['score']} | {pv} | "
                  f"{o['agreement']:.2f} | {o['speedup']:.2f}x | {o['verdict']} | {tr} |")
            w("")
        if unlock:
            w(f"*Unlock move:* {unlock}.")
            w("")
    if red["hardware_candidates"]:
        w("### Hardware-only candidates (proposed, not simulated)")
        w("")
        for h in red["hardware_candidates"]:
            w(f"- `{h['transform']}` on `{h['from']}` -> `{h['point']}` (targets {', '.join(h['targets'])}): "
              f"{'; '.join(h['why_not_simulated'])}")
        w("")
    w("## 2. Robustness: the same search from every starting point")
    w("")
    w("| Start | Sims | Best trusted found | Cycles | Within 5% of best | Stop reason |")
    w("|---|---|---|---|---|---|")
    for r in d["robustness"]:
        w(f"| {fmt_pt(r['start'])} | {r['sims']} | {fmt_pt(r['best']) if r['best'] else '-'} | "
          f"{r['best_cycles']:,} | {'yes' if r['within_5pct'] else 'no'} | {r['stop']} |")
    w("")
    w("## 3. Same simulation budget: guided vs unguided vs random (start: v1)")
    w("")
    s = d["budget_comparison"]
    w(f"Unguided local search uses the same transformation menu and acceptance rule as the guided search but "
      f"ignores counters (random neighbour order); random search samples design points uniformly. "
      f"Both averaged over {a.seeds} seeds. Each simulation costs 1 to 45 minutes of CPU time here, so "
      "*simulations to reach a good kernel* is the cost that matters.")
    w("")
    w("| Strategy | Runs that reached within 5% of best | Mean sims to reach it (runs that did) | Median | Mean best cycles at full budget |")
    w("|---|---|---|---|---|")
    for k, nm in (("counter_guided", "Counter-guided"), ("unguided_local", "Unguided local search"),
                  ("random", "Random search")):
        x = s[k]
        w(f"| {nm} | {100 * x['p_reach']:.0f}% | {x['mean_sims'] if x['mean_sims'] is None else round(x['mean_sims'], 1)} | "
          f"{x['median_sims']} | {x['final_mean_cycles']:,.0f} |")
    w("")
    w("![Simulated: best speedup found vs simulation budget](budget_curve.png)")
    w("")
    w("## 4. Signature audit: were the declared counter signatures right?")
    w("")
    w(f"A counter 'moved' if it changed by more than {100 * SIGN_TOL:.0f}%. Observed signs are listed in the "
      "order the transformation was applied during the rediscovery run.")
    w("")
    w("| Transformation | Counter | Declared | Observed | Status |")
    w("|---|---|---|---|---|")
    for t, cs in d["signature_audit"].items():
        for k, (want, seen, status) in cs.items():
            w(f"| `{t}` | {k} | {want} | {seen} | {status} |")
    w("")
    w("## Method (as run)")
    w("")
    w(f"- Bottleneck rules (`autotune/diagnose.py`), fixed before any search was run:")
    for k, v in THRESHOLDS.items():
        w(f"  - `{k}`: {v}")
    w(f"- Transformation menu (`autotune/transforms.py`): {', '.join(f'`{t.name}`' for t in MENU)}.")
    w(f"- Candidate score = sum of targeted bottleneck severities x Beta-posterior confidence of "
      f"(bottleneck, transformation); beam {a.beam}; a step must improve cycles by >= {100 * MIN_GAIN:.0f}% "
      f"with a trusted result; 'moved as declared' = at least {AGREE:.0%} of declared counters match.")
    w("- Spills are found statically (`ptxas -v` for sm_80), because the simulator executes PTX and cannot see them.")
    w("")
    w("## What this does not show yet")
    w("")
    w("- Whether the simulated ranking holds on a real A100: run `autotune/validate_hw.py` (Measured).")
    w("- Generalization beyond attention: the menu and thresholds were written for this kernel family.")
    w("- One workload only. Re-run `groundtruth` + `experiments` with other `--BH/--N/--causal` to test.")
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
