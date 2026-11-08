"""Hardware validation of the frozen autotune-v1 against its pre-registered predictions (Measured).

    python -m autotune.validate_hw                     # timing + Nsight Compute counters if permitted
    python -m autotune.validate_hw --profile off       # timing only
    python -m autotune.validate_hw --rescore           # re-score saved measurements (no GPU needed)

Steps
  1. check the frozen files against results/autotune/preregistration/manifest.json
  2. build sim/harness natively, time all 23 design points (CUDA events, median of --iters)
  3. optionally profile every point with Nsight Compute through the frozen sim->hardware mapping
     (autotune/hw_counters.py). No ncu, or no counter permission: counters are UNOBSERVABLE and
     timing validation still runs. Registers stay observable from the native ptxas build.
  4. score predictions.json, keeping three questions separate:
       performance fidelity  does simulation rank configurations correctly?
       mechanism fidelity    do the predicted counter changes happen on hardware?
       decision fidelity     would the frozen optimizer have made useful decisions on hardware data?
Writes results/autotune/measured/{hw_measurements.json, scores.json, validation.md}. Every number in
those files is Measured; predictions are read from the pre-registration and never recomputed.

Nothing here changes the optimizer. The frozen policy is rerun unchanged on hardware counters
(including its trust check, which still rejects spilling points: it is the experimental subject).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict

from .backends import REPO, Backend, StaticInfo
from .counters import Counters
from .diagnose import diagnose
from .freeze import COUNTER_TOL, PERF_TOL, PREREG, perf_dir, sign, verify
from .hw_counters import MAPPING, UNOBSERVABLE, aggregate, ncu_path, profile
from .search import counter_guided
from .space import DesignPoint, Workload, all_points, make

OUT = os.path.join(REPO, "results", "autotune", "measured")
ARROW = {"+": "↑", "-": "↓", "0": "=", "?": "?", None: "-"}


# ----------------------------------------------------------------------------------- statistics
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = a[i] - a[j], b[i] - b[j]
            s += (x * y > 0) - (x * y < 0)
    return s / (n * (n - 1) / 2) if n > 1 else float("nan")


# ----------------------------------------------------------------------------------- measurement
def sm_count():
    q = "import torch;p=torch.cuda.get_device_properties(0);print(p.multi_processor_count,p.major,p.minor,p.name)"
    out = subprocess.run(["python", "-c", q], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if out.returncode == 0 and out.stdout.strip():
        sms, ma, mi, *name = out.stdout.split()
        return int(sms), f"{ma}{mi}", " ".join(name)
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    name, cap = (out.stdout.strip().split(", ") + ["8.0"])[:2] if out.stdout.strip() else ("unknown", "8.0")
    return 108, cap.replace(".", ""), name


def build(arch: str) -> str:
    exe = os.path.join(REPO, "sim", f"harness_sm{arch}")
    subprocess.run(["nvcc", "-O3", "-std=c++17", f"-arch=sm_{arch}", "-o", exe,
                    os.path.join(REPO, "sim", "harness.cu")], check=True)
    return exe


def time_point(exe, p, wl, iters):
    import re
    env = dict(os.environ, HARNESS_ITERS=str(iters))
    proc = subprocess.run([exe, p.harness_spec(), str(wl.BH), str(wl.N), str(wl.causal)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    st = re.findall(r"RESULT .* status=(\w+)", proc.stdout)
    t = re.findall(r"TIME kernel=\S+ BH=\d+ N=\d+ causal=\d median_ms=([\d.]+) min_ms=([\d.]+)", proc.stdout)
    return (st[-1] if st else "ERROR"), (float(t[-1][0]) if t else None), (float(t[-1][1]) if t else None)


def measure(a) -> dict:
    sms, arch, name = sm_count()
    scale = sms / 14 if a.bh_scale == "auto" else float(a.bh_scale)
    wl = Workload(max(1, round(42 * scale)), 128, 0)
    exe = build(arch)
    static = StaticInfo.from_ptxas(arch=f"sm_{arch}",
                                   cache=os.path.join(OUT, f"static_sm{arch}.json"))
    want_prof = a.profile != "off" and ncu_path() is not None
    prof_status = "disabled" if a.profile == "off" else ("no_ncu" if not ncu_path() else "pending")
    if want_prof:  # one probe decides whether counters are available at all
        probe = profile(exe, make("fused_tc").harness_spec(), 2, 64, 0)
        prof_status = probe["status"]
        want_prof = prof_status == "ok"
        if not want_prof:
            print(f"Nsight Compute unavailable ({prof_status}); timing only. {probe['log'][-400:]}")
    points = {}
    for p in all_points(include_blind=True):
        status, med, mn = time_point(exe, p, wl, a.iters)
        rec = {"status": status, "median_ms": med, "min_ms": mn, "ncu": None, "dropped_metrics": []}
        if want_prof and status == "PASS":
            pr = profile(exe, p.harness_spec(), wl.BH, wl.N, wl.causal)
            rec["ncu_status"] = pr["status"]
            rec["dropped_metrics"] = pr["dropped_metrics"]
            rec["ncu_kernels"] = pr["kernels"]
            rec["ncu"] = aggregate(pr["kernels"]) if pr["status"] == "ok" else None
        points[p.key()] = rec
        print(f"{p.key():32} {status:5} {med if med is not None else '-':>10} ms  ncu={rec.get('ncu_status', '-')}")
    return {"gpu": name, "sms": sms, "arch": arch, "workload": {"BH": wl.BH, "N": wl.N, "causal": wl.causal},
            "bh_scale": scale, "iters": a.iters, "profiling": prof_status, "points": points,
            "static_table": static.table}


# ----------------------------------------------------------------------------------- hardware counters
def hw_values(meas: dict, sim_cache: dict, key: str) -> dict:
    """Mapped hardware value per simulator counter for one point (None = UNOBSERVABLE)."""
    rec = meas["points"].get(key, {})
    vals = dict(rec.get("ncu") or {k: None for k in MAPPING})
    vals["cycles"] = rec.get("median_ms")
    if vals.get("regs") is None:  # static fallback: native ptxas registers of the dominant kernel
        dom = (sim_cache.get(key) or {}).get("dominant_kernel")
        s = meas.get("static_table", {}).get(dom or "", {})
        vals["regs"] = s.get("regs")
        vals["regs_source"] = "ptxas (static)" if s else None
    return vals


def spill_bytes(meas, sim_cache, key):
    dom = (sim_cache.get(key) or {}).get("dominant_kernel")
    return meas.get("static_table", {}).get(dom or "", {}).get("spill_stores", 0)


def hw_dir(v0, v1):
    if v0 is None or v1 is None:
        return UNOBSERVABLE
    if v0 == 0:
        return "0" if v1 == 0 else "+"
    return sign((v1 - v0) / abs(v0), COUNTER_TOL)


def match(pred, obs):
    if obs == UNOBSERVABLE:
        return UNOBSERVABLE
    if pred in (None, "?"):
        return "NO_CLAIM"
    return "MATCH" if pred == obs else "MISMATCH"


def hw_counters(vals: dict, p: DesignPoint, wl: Workload, sms: int, spill: int) -> Counters | None:
    """Hardware values in the simulator's counter vocabulary, so the frozen rules can read them."""
    need = ["cycles", "occupancy", "regs", "cta_per_sm", "kernel_launches", "l2_accesses", "thread_insn",
            "warp_insn", "shmem_insn", "shmem_bank_conflicts", "scoreboard_frac", "global_load_insn",
            "dram_sectors", "issue_util"]
    if any(vals.get(k) is None for k in need) or vals.get("cta_limiter") is None:
        return None
    raw = {k: vals[k] for k in need if k not in ("dram_sectors", "issue_util", "scoreboard_frac")}
    raw["cycles"] = vals["cycles"] * 1e6  # ns; only ratios are used
    raw["cta_per_sm"] = int(vals["cta_per_sm"])
    raw["kernel_launches"] = int(vals["kernel_launches"])
    raw["cta_limiter"] = vals["cta_limiter"]
    raw["hw_spill_bytes"] = spill
    blocks = wl.blocks(p)
    der = {"dram_sectors": vals["dram_sectors"], "l2_traffic_ratio": vals["l2_accesses"] / wl.compulsory_sectors(),
           "insn_per_flop": vals["thread_insn"] / wl.flops(), "issue_util": vals["issue_util"] / 100,
           "scoreboard_frac": vals["scoreboard_frac"] / 100,
           "smem_conflict_rate": vals["shmem_bank_conflicts"] / vals["shmem_insn"] if vals["shmem_insn"] else 0.0,
           "blocks": blocks, "waves": blocks / (sms * raw["cta_per_sm"]) if blocks and raw["cta_per_sm"] else 0.0}
    return Counters("PASS", 0.0, raw, der)


class HardwareBackend(Backend):
    name = "hardware"

    def __init__(self, table: dict):
        super().__init__()
        self.table = table

    def evaluate(self, p, wl):
        if p.key() not in self.calls:
            self.calls.append(p.key())
        return self.table[p.key()]


# ----------------------------------------------------------------------------------- scoring
def score(meas: dict, pred: dict, sim_cache: dict, frozen_ok: list) -> dict:
    pts = meas["points"]
    ms = {k: r["median_ms"] for k, r in pts.items() if r["status"] == "PASS" and r["median_ms"] is not None}
    rank = {r["point"]: r for r in pred["simulated_ranking"]}
    sc = {"frozen_check": frozen_ok or "frozen files match manifest", "hypotheses": {}}
    H = sc["hypotheses"]

    # ---------------- performance fidelity
    both = [k for k in rank if k in ms]
    s_cyc, m = [rank[k]["sim_cycles"] for k in both], [ms[k] for k in both]
    rho, tau = (spearman(s_cyc, m), kendall(s_cyc, m)) if len(both) > 2 else (float("nan"),) * 2
    by_sim = sorted(both, key=lambda k: rank[k]["sim_cycles"])
    by_hw = sorted(both, key=lambda k: ms[k])
    trusted_sim = [k for k in by_sim if rank[k]["trusted"]]
    sel = trusted_sim[0] if trusted_sim else None
    best_all = min(ms, key=ms.get) if ms else None
    perf = {"n": len(both), "spearman": rho, "kendall": tau,
            "top1_sim_trusted": sel, "top1_hw_among_simulated": by_hw[0] if by_hw else None,
            "top1_agree": bool(by_hw) and sel == by_hw[0],
            "top3_overlap": len(set(by_sim[:3]) & set(by_hw[:3])),
            "regret": ms[sel] / ms[by_hw[0]] if sel and by_hw else None,
            "best_all_23": best_all, "ranking": [{"point": k, "sim_rank": by_sim.index(k) + 1,
                                                 "hw_rank": by_hw.index(k) + 1, "sim_cycles": rank[k]["sim_cycles"],
                                                 "measured_ms": ms[k], "trusted": rank[k]["trusted"]}
                                                for k in by_hw]}
    sc["performance_fidelity"] = perf
    H["H1"] = {"value": rho, "result": "supported" if rho >= 0.8 else "partial" if rho >= 0.5 else "refuted"}
    H["H2"] = {"value": best_all, "result": "supported" if best_all == "fused_tc" else "refuted"}
    if "fused/br64/bc32" in ms and "tiled" in ms:
        r = ms["fused/br64/bc32"] / ms["tiled"]
        H["H3"] = {"value": r, "result": "simulator ordering holds" if r > 1 + PERF_TOL else
                   "reversed" if r < 1 - PERF_TOL else "tie"}
    else:
        H["H3"] = {"value": None, "result": "inconclusive (missing measurement)"}
    if "fused_tc" in ms:
        r = ms["fused_tc"] / min(ms.values())
        H["H6"] = {"value": r, "result": "supported" if r <= 1.05 else "refuted"}

    # ---------------- mechanism fidelity
    trans_scores = []
    for t in pred["transitions"]:
        p0, p1 = t["parent"], t["child"]
        if p0 not in ms or p1 not in ms:
            continue
        v0, v1 = hw_values(meas, sim_cache, p0), hw_values(meas, sim_cache, p1)
        hw_perf = perf_dir(ms[p0] / ms[p1])
        rows = {}
        for k, m_ in MAPPING.items():
            if k == "cycles":
                continue
            obs = hw_dir(v0.get(k), v1.get(k))
            sim_p = t.get("sim_counter_direction", {}).get(k) if t["sim_verifiable"] else None
            dec_p = t["declared_signature"].get(k)
            rows[k] = {"hw": obs, "sim_pred": sim_p, "declared": dec_p,
                       "sim_match": match(sim_p, obs) if t["sim_verifiable"] else "NO_CLAIM",
                       "declared_match": match(dec_p, obs), "equivalence": m_["equivalence"]}
        rows["performance"] = {"hw": hw_perf, "sim_pred": t["predicted_performance"] if t["sim_verifiable"] else None,
                               "declared": {"-": "faster", "+": "slower", "0": "same"}.get(
                                   t["declared_signature"].get("cycles"), None),
                               "equivalence": "performance"}
        rp = rows["performance"]
        rp["sim_match"] = ("MATCH" if rp["sim_pred"] == hw_perf else "MISMATCH") if rp["sim_pred"] else "NO_CLAIM"
        rp["declared_match"] = ("MATCH" if rp["declared"] == hw_perf else "MISMATCH") if rp["declared"] else "NO_CLAIM"
        trans_scores.append({"transform": t["transform"], "parent": p0, "child": p1,
                             "sim_verifiable": t["sim_verifiable"], "child_trusted": t.get("child_trusted"),
                             "sim_speedup": t["sim_speedup"], "hw_speedup": ms[p0] / ms[p1], "rows": rows})
    sc["transitions"] = trans_scores

    def acc(rows_iter, which):
        c = defaultdict(int)
        for r in rows_iter:
            c[r[which]] += 1
        n = c["MATCH"] + c["MISMATCH"]
        return {"match": c["MATCH"], "mismatch": c["MISMATCH"], "unobservable": c[UNOBSERVABLE],
                "no_claim": c["NO_CLAIM"], "accuracy": c["MATCH"] / n if n else None}

    per_counter = {}
    for k in list(MAPPING) + ["performance"]:
        if k == "cycles":
            continue
        rs = [ts["rows"][k] for ts in trans_scores if k in ts["rows"]]
        per_counter[k] = {"equivalence": MAPPING[k]["equivalence"] if k in MAPPING else "performance",
                          "label": MAPPING[k]["label"] if k in MAPPING else "Performance direction",
                          "simulator": acc(rs, "sim_match"), "declared": acc(rs, "declared_match")}
    per_class = {}
    for cls in ("same_definition", "approximate", "proxy", "performance"):
        rs = [ts["rows"][k] for ts in trans_scores for k in ts["rows"] if ts["rows"][k]["equivalence"] == cls]
        per_class[cls] = {"simulator": acc(rs, "sim_match"), "declared": acc(rs, "declared_match")}
    per_tf = {}
    for ts in trans_scores:
        d = per_tf.setdefault(ts["transform"], defaultdict(list))
        for k, r in ts["rows"].items():
            d[k].append(r)
    per_tf = {t: {k: {"declared": rs[0]["declared"], "simulator": acc(rs, "sim_match"),
                      "declared_acc": acc(rs, "declared_match"), "n": len(rs)} for k, rs in d.items()}
              for t, d in per_tf.items()}
    sc["mechanism_fidelity"] = {"per_counter": per_counter, "per_equivalence": per_class, "per_transformation": per_tf}
    a_same = per_class["same_definition"]["simulator"]["accuracy"]
    a_app = per_class["approximate"]["simulator"]["accuracy"]
    a_prx = per_class["proxy"]["simulator"]["accuracy"]
    if None in (a_same, a_app, a_prx):
        H["H7"] = {"value": [a_same, a_app, a_prx], "result": "inconclusive (counters unobservable)"}
    else:
        H["H7"] = {"value": [a_same, a_app, a_prx],
                   "result": "supported" if a_same > a_app and a_same > a_prx else "refuted"}

    # H4: register caps (sim speedup of capping vs measured)
    pairs = []
    for t in trans_scores:
        if t["transform"] == "reg_cap_128" and t["sim_verifiable"]:
            pairs.append({"pair": f"{t['parent']} -> {t['child']}", "sim": t["sim_speedup"], "hw": t["hw_speedup"],
                          "overstated": t["hw_speedup"] < t["sim_speedup"] - 0.02})
    k_ = sum(p["overstated"] for p in pairs)
    H["H4"] = {"value": pairs, "result": ("inconclusive (fewer than 6 pairs measured)" if len(pairs) < 6 else
                                          "supported" if k_ >= 4 else "refuted" if k_ <= 2 else "inconclusive")}
    # H8: double buffering (rule only)
    db = [{"pair": f"{t['parent']} -> {t['child']}", "hw": t["hw_speedup"], "faster": t["hw_speedup"] > 1 + PERF_TOL}
          for t in trans_scores if t["transform"] == "double_buffer" and "/r128" not in t["parent"]]
    k_ = sum(d["faster"] for d in db)
    H["H8"] = {"value": db, "result": ("inconclusive (fewer than 4 pairs measured)" if len(db) < 4 else
                                       "supported" if k_ >= 3 else "refuted" if k_ <= 1 else "inconclusive")}

    # ---------------- decision fidelity
    steps = []
    path = pred["search"]["path"]
    for a_, b_ in zip(path[:-1], path[1:]):
        t = next((t for t in pred["transitions"] if t["parent"] == a_ and t["child"] == b_), None)
        if t and a_ in ms and b_ in ms:
            hp = perf_dir(ms[a_] / ms[b_])
            steps.append({"parent": a_, "child": b_, "transform": t["transform"],
                          "predicted": t["predicted_performance"], "sim_speedup": t["sim_speedup"],
                          "hw_speedup": ms[a_] / ms[b_], "hw": hp, "match": hp == t["predicted_performance"]})
    improving_ok = all(s["match"] for s in steps if s["predicted"] == "faster")
    H["H5"] = {"value": steps, "result": "inconclusive (missing steps)" if len(steps) < len(path) - 1 else
               "supported" if all(s["match"] for s in steps) else "partial" if improving_ok else "refuted"}
    decisions = []
    for o in pred["search"]["decisions"]:
        if o["parent"] in ms and o["child"] in ms:
            hw_s = ms[o["parent"]] / ms[o["child"]]
            decisions.append({"step": o["step"], "transform": o["transform"], "parent": o["parent"], "child": o["child"],
                              "sim_speedup": o["speedup"], "hw_speedup": hw_s, "trusted": o["trusted"],
                              "sim_verdict": o["verdict"],
                              "same_direction": perf_dir(o["speedup"]) == perf_dir(hw_s)})

    # bottleneck classification: frozen rules on simulator counters vs on hardware counters
    sms = meas["sms"]
    hwwl = Workload(meas["workload"]["BH"], meas["workload"]["N"], meas["workload"]["causal"])
    table, missing = {}, []
    for k in ms:
        p = DesignPoint.from_key(k)
        c = hw_counters(hw_values(meas, sim_cache, k), p, hwwl, sms, spill_bytes(meas, sim_cache, k))
        if c is None:
            missing.append(k)
        else:
            table[k] = c
    classification, h9, greedy_hw = [], None, None
    if table:
        from .backends import ReplayBackend
        rb = ReplayBackend()
        simwl = Workload()
        for k in sorted(table):
            if not rb.has(DesignPoint.from_key(k), simwl):
                continue
            p = DesignPoint.from_key(k)
            s_set = {b.name for b in diagnose(rb.evaluate(p, simwl), p)}
            h_set = {b.name for b in diagnose(table[k], p)}
            classification.append({"point": k, "sim": sorted(s_set), "hw": sorted(h_set),
                                   "jaccard": len(s_set & h_set) / len(s_set | h_set) if s_set | h_set else 1.0})
        if "naive" in table:
            hb = HardwareBackend(table)
            res, _ = counter_guided(make("naive"), hwwl, hb, budget=len(table))
            h9 = {"path": res.path, "best": res.best[1], "sims": len(res.evaluated), "stop": res.stop_reason,
                  "outcomes": res.outcomes, "diagnoses": res.diagnoses}
            hb = HardwareBackend(table)
            g, _ = counter_guided(make("naive"), hwwl, hb, budget=len(table), max_unlock=0)
            greedy_hw = {"path": g.path, "best": g.best[1], "stop": g.stop_reason}
    if h9 is None:
        H["H9"] = {"value": None, "result": "inconclusive (Nsight counters unavailable)" if not table else
                   "inconclusive (naive not profiled)"}
    else:
        H["H9"] = {"value": h9["path"], "result": "supported" if h9["path"][-1] == "fused_tc" else "refuted"}
    sc["decision_fidelity"] = {"path_steps": steps, "decisions": decisions, "classification": classification,
                               "hw_counter_points_missing": missing, "search_on_hardware": h9,
                               "greedy_on_hardware": greedy_hw}
    return sc


# ----------------------------------------------------------------------------------- report
def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def report(meas, pred, sc) -> str:
    L = []
    w = L.append
    wl = meas["workload"]
    w("# Hardware validation of autotune-v1 (Measured)")
    w("")
    w(f"GPU: {meas['gpu']} ({meas['sms']} SMs, sm_{meas['arch']}). Workload: BH={wl['BH']}, N={wl['N']}, "
      f"causal={wl['causal']}, d=64, FP16 (simulated BH=42 on a 14-SM slice, scaled x{meas['bh_scale']:.2f}). "
      f"Time: median of {meas['iters']} launches, CUDA events. Nsight Compute: **{meas['profiling']}**.")
    w("")
    w(f"Frozen subject: commit `{pred['frozen_commit'][:12]}`. Pre-registration check: "
      f"**{sc['frozen_check'] if isinstance(sc['frozen_check'], str) else 'MISMATCH: ' + '; '.join(sc['frozen_check'])}**. "
      "Predictions are read from `results/autotune/preregistration/predictions.json`, never recomputed.")
    w("")
    w("## Pre-registered hypotheses")
    w("")
    w("| ID | Question | Result | Value |")
    w("|---|---|---|---|")
    hyp = {h["id"]: h for h in pred["hypotheses"]}
    for hid in sorted(sc["hypotheses"], key=lambda x: int(x[1:])):
        h = sc["hypotheses"][hid]
        v = h["value"]
        vs = (f"{v:.3f}" if isinstance(v, float) else " -> ".join(v) if hid == "H9" and v else
              f"{sum(x.get('overstated', x.get('faster', x.get('match', False))) for x in v)} of {len(v)}"
              if isinstance(v, list) and v and isinstance(v[0], dict) else
              ", ".join(pct(x) for x in v) if hid == "H7" else str(v))
        w(f"| {hid} | {hyp[hid]['question']} | **{h['result']}** | {vs} |")
    w("")

    # 1. performance
    pf = sc["performance_fidelity"]
    w("## 1. Performance fidelity: does simulation rank configurations correctly?")
    w("")
    w(f"- Points compared: {pf['n']}; Spearman rho **{pf['spearman']:.3f}**, Kendall tau {pf['kendall']:.3f}")
    w(f"- Simulator's best trusted point `{pf['top1_sim_trusted']}`; measured best among simulated "
      f"`{pf['top1_hw_among_simulated']}` (agree: {'yes' if pf['top1_agree'] else 'no'}); top-3 overlap "
      f"{pf['top3_overlap']} of 3; regret {pf['regret']:.2f}x" if pf["regret"] else "- no ranking")
    w(f"- Measured best of all 23 points (including double buffering): `{pf['best_all_23']}`")
    w("")
    w("| Design point | Simulated cycles | Sim rank | Measured ms | HW rank | Trusted in sim |")
    w("|---|---|---|---|---|---|")
    for r in pf["ranking"]:
        w(f"| {r['point']} | {r['sim_cycles']:,} | {r['sim_rank']} | {r['measured_ms']:.4f} | {r['hw_rank']} | "
          f"{'yes' if r['trusted'] else 'no'} |")
    w("")

    # 2. mechanism
    mf = sc["mechanism_fidelity"]
    w("## 2. Mechanism fidelity: do the predicted counter changes happen on hardware?")
    w("")
    w(f"Each transition is scored twice: against the **simulator's** observed direction and against the "
      f"transformation's **declared** signature. A counter moved if it changed by more than "
      f"{100 * COUNTER_TOL:.0f}%; performance by more than {100 * PERF_TOL:.0f}%. Unobservable hardware values "
      "are counted separately, never as mismatches.")
    w("")
    w("### Prediction accuracy by counter")
    w("")
    w("| Counter | Mapping | Simulator accuracy | (match / mismatch / unobservable) | Declared-signature accuracy | (match / mismatch) |")
    w("|---|---|---|---|---|---|")
    for k, r in mf["per_counter"].items():
        s, d = r["simulator"], r["declared"]
        w(f"| {r['label']} | {r['equivalence']} | **{pct(s['accuracy'])}** | {s['match']} / {s['mismatch']} / "
          f"{s['unobservable']} | {pct(d['accuracy'])} | {d['match']} / {d['mismatch']} |")
    w("")
    w("### By mapping equivalence (H7)")
    w("")
    w("| Equivalence | Simulator accuracy | Declared accuracy | Unobservable |")
    w("|---|---|---|---|")
    for cls, r in mf["per_equivalence"].items():
        w(f"| {cls} | {pct(r['simulator']['accuracy'])} | {pct(r['declared']['accuracy'])} | "
          f"{r['simulator']['unobservable']} |")
    w("")
    w("### Transitions on the search path")
    w("")
    path = pred["search"]["path"]
    for ts in sc["transitions"]:
        if (ts["parent"], ts["child"]) not in set(zip(path[:-1], path[1:])):
            continue
        w(f"**`{ts['transform']}`**: `{ts['parent']}` -> `{ts['child']}` "
          f"(simulated {ts['sim_speedup']:.2f}x, measured {ts['hw_speedup']:.2f}x)")
        w("")
        w("| Counter | Simulator predicted | A100 | Match | Declared | Match |")
        w("|---|---|---|---|---|---|")
        nm = nn = 0
        for k, r in ts["rows"].items():
            lab = MAPPING[k]["label"] if k in MAPPING else "Performance"
            sym = (lambda x: x if k == "performance" else ARROW.get(x, x))
            mk = {"MATCH": "✓", "MISMATCH": "✗", UNOBSERVABLE: "unobservable", "NO_CLAIM": ""}
            w(f"| {lab} | {sym(r['sim_pred'])} | {sym(r['hw'])} | {mk[r['sim_match']]} | "
              f"{sym(r['declared']) if r['declared'] else ''} | {mk[r['declared_match']]} |")
            nm += r["sim_match"] == "MATCH"
            nn += r["sim_match"] in ("MATCH", "MISMATCH")
        w("")
        w(f"Simulator prediction accuracy: {nm}/{nn}")
        w("")
    w("### All transformations (declared signature vs hardware, all transitions)")
    w("")
    w("| Transformation | Counter | Declared | Simulator acc. | Declared acc. | Transitions |")
    w("|---|---|---|---|---|---|")
    for t, d in mf["per_transformation"].items():
        for k, r in d.items():
            if r["declared"] is None and r["simulator"]["accuracy"] is None:
                continue
            lab = MAPPING[k]["label"] if k in MAPPING else "Performance"
            w(f"| `{t}` | {lab} | {ARROW.get(r['declared'], r['declared']) if r['declared'] else ''} | "
              f"{pct(r['simulator']['accuracy'])} | {pct(r['declared_acc']['accuracy'])} | {r['n']} |")
    w("")

    # 3. decisions
    df = sc["decision_fidelity"]
    w("## 3. Decision fidelity: would the frozen optimizer have decided well on hardware data?")
    w("")
    w("### Every decision the simulated search judged")
    w("")
    w("| Step | Transformation | Parent -> child | Simulated | Measured | Same direction | Trusted in sim |")
    w("|---|---|---|---|---|---|---|")
    for d in df["decisions"]:
        w(f"| {d['step']} | `{d['transform']}` | `{d['parent']}` -> `{d['child']}` | {d['sim_speedup']:.2f}x | "
          f"{d['hw_speedup']:.2f}x | {'yes' if d['same_direction'] else '**no**'} | {'yes' if d['trusted'] else 'no'} |")
    if df["decisions"]:
        w("")
        w(f"Direction held for {sum(d['same_direction'] for d in df['decisions'])} of {len(df['decisions'])}.")
    w("")
    w("### Bottleneck classification: frozen rules on simulator vs hardware counters")
    w("")
    if df["classification"]:
        w("| Point | Simulator diagnosis | Hardware diagnosis | Agreement (Jaccard) |")
        w("|---|---|---|---|")
        for c in df["classification"]:
            w(f"| {c['point']} | {', '.join(c['sim']) or '-'} | {', '.join(c['hw']) or '-'} | {c['jaccard']:.2f} |")
        w("")
        w(f"Mean agreement: {sum(c['jaccard'] for c in df['classification']) / len(df['classification']):.2f}")
    else:
        w("Unobservable: hardware counters were not collected.")
    w("")
    w("### The frozen search, rerun on hardware counters and hardware time (H9)")
    w("")
    h9 = df["search_on_hardware"]
    if h9:
        w(f"Path: {' -> '.join(h9['path'])}; best {h9['best']}; {h9['sims']} measurements; stop: {h9['stop']}.")
        w("")
        g = df["greedy_on_hardware"]
        w(f"Pure greedy (no unlock) on hardware: {' -> '.join(g['path'])}; best {g['best']}. "
          "Compare with H3: in simulation greedy stopped at v2.")
        w("")
        w("The frozen trust check still rejects spilling points on hardware, and double-buffered points are "
          "still never evaluated: the subject is run exactly as frozen.")
    else:
        w("Not run: " + ("hardware counters missing for " + ", ".join(df["hw_counter_points_missing"])
                         if df["hw_counter_points_missing"] else "Nsight Compute unavailable") + ".")
    w("")
    w("## What changes now")
    w("")
    w("Nothing in autotune-v1. Failures above are results. Changes motivated by these measurements belong in "
      "autotune-v2, reported against this file.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh-scale", default="auto", help="'auto' = SMs/14, or a number")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--profile", choices=["auto", "off"], default="auto")
    ap.add_argument("--rescore", action="store_true", help="score saved hw_measurements.json without a GPU")
    a = ap.parse_args()
    man = json.load(open(os.path.join(PREREG, "manifest.json")))
    pred = json.load(open(os.path.join(PREREG, "predictions.json")))
    frozen = verify(man)
    print("frozen files match manifest" if not frozen else "WARNING, frozen files changed: " + "; ".join(frozen))
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, "hw_measurements.json")
    if a.rescore:
        if not os.path.exists(mpath):
            raise SystemExit(f"--rescore: no saved measurements at {mpath}; run on a GPU first")
        meas = json.load(open(mpath))
    else:
        meas = measure(a)
        json.dump(meas, open(mpath, "w"), indent=1)
    sim_cache = {v["point"]: v["counters"] for v in
                 json.load(open(os.path.join(REPO, "results", "autotune", "sim_cache.json"))).values()}
    sc = score(meas, pred, sim_cache, frozen)
    json.dump(sc, open(os.path.join(OUT, "scores.json"), "w"), indent=1, default=str)
    rep = report(meas, pred, sc)
    open(os.path.join(OUT, "validation.md"), "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
