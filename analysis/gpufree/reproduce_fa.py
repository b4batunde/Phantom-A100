"""Reproduce published FlashAttention results without a GPU.

    cd analysis && python -m gpufree.reproduce_fa        -> analysis/fa_reproduction.md (+ .csv)

Every claim is checked against a model built only from datasheet numbers and operation /
memory-pass counts. A claim is "reproduced" when the measured value falls inside the
model's bounds; how tight the bounds are is reported alongside.

Published values (quoted from the papers):
  FA1 = Dao et al. 2022, arXiv:2205.14135   FA2 = Dao 2023, arXiv:2307.08691
"""
from __future__ import annotations

import csv
import os

from .model import A100_40GB, A100_80GB, fa2_bounds

FP16 = 2  # bytes; the papers' benchmarks use FP16/BF16 inputs

# ----------------------------------------------------------------------------- published numbers
FA1_FIG2 = dict(  # FA1 Figure 2 (left): GPT-2 medium, seq 1024, head dim 64, 16 heads, batch 64, fwd + bwd, A100
    B=64, H=16, N=1024, d=64,
    std_hbm_gb=40.3, std_ms=41.7, std_gflops=66.6,
    fa_hbm_gb=4.4, fa_ms=7.3, fa_gflops=75.2,
)
FA2_FWD_D64 = {  # FA2 Fig. 5, forward, head dim 64, A100 80GB, values read from the figure's text layer (approximate)
    False: [178, 191, 193, 192, 192, 192],   # non-causal, seq 512 .. 16k
    True: [115, 146, 167, 177, 181, 183],    # causal
}
FA2_HEADLINE_TFLOPS = 230  # "up to 230 TFLOPs/s, 73% of the theoretical maximum" (FA2 Section 4)
FA2_HEADLINE_PCT = 73
FA2_VS_STANDARD = (3, 10)  # "3-10x faster than a standard attention implementation" (FA2 Section 4)


# ----------------------------------------------------------------------------- HBM pass counting
def standard_attention_hbm(B, H, N, d, variant: str) -> float:
    """Bytes moved to/from HBM by standard (unfused) attention, forward + backward.

    "minimal":   FA1's Algorithm 0 (forward) and Algorithm 3 (backward) exactly as written:
                 fwd  write S, read S, write P, read P                          -> 4 N^2 passes
                 bwd  read P (dV), write dP, read P+dP & write dS, read dS x2   -> 7 N^2 passes
    "framework": a typical framework graph where scale, causal mask, softmax and dropout are
                 separate elementwise kernels, each reading and writing the N x N tensor
                 (dropout also writes a 1-byte mask):
                 fwd  1 + 2 + 2 + 2 + 2.5 + 1 = 10.5 passes,  bwd 1 + 1 + 2.5 + 3 + 2 + 2 + 2 = 13.5
    Both add 12 N x d tensor passes (fwd: Q K V read, O write; bwd: dO twice, Q K V, dQ dK dV).
    """
    n2 = {"minimal": 4 + 7, "framework": 10.5 + 13.5}[variant]
    return (n2 * N * N + 12 * N * d) * B * H * FP16


def flash_attention_hbm(B, H, N, d, block: int | None) -> float:
    """Bytes moved by FlashAttention, forward + backward.

    block=None: compulsory traffic only (every input read once, every output written once):
                fwd Q K V read + O write (4), bwd Q K V O dO read + dQ dK dV write (8) -> 12 N x d passes.
    block=Bc:   FA1 Algorithm 1 (fwd) and Algorithm 4 (bwd) as written, with no cache reuse.
                Tc = N / Bc outer iterations.
                fwd: K, V once (2) + per outer iteration read Q, read+write O (3 Tc)
                bwd: K, V once + dK, dV written once (4) + per outer iteration read Q, O, dO, dQ and write dQ (5 Tc)
                This is Theorem 2's Theta(N^2 d^2 / M) term with Bc ~ M / 4d.
    """
    if block is None:
        nd_passes = 12
    else:
        tc = N / block
        nd_passes = (2 + 3 * tc) + (4 + 5 * tc)
    return nd_passes * N * d * B * H * FP16


# ----------------------------------------------------------------------------- the checks
def check_fa1_fig2():
    p = FA1_FIG2
    B, H, N, d = p["B"], p["H"], p["N"], p["d"]
    gpu = A100_40GB  # FA1 Fig. 1 quotes HBM 1.5 TB/s, 40 GB
    std = {v: standard_attention_hbm(B, H, N, d, v) / 1e9 for v in ("minimal", "framework")}
    fa_floor = flash_attention_hbm(B, H, N, d, None) / 1e9
    fa_alg = {bc: flash_attention_hbm(B, H, N, d, bc) / 1e9 for bc in (64, 128, 256, 512)}
    rows = [
        dict(claim="FA1 Fig.2: standard attention HBM R/W", published=f"{p['std_hbm_gb']} GB",
             model=f"{std['minimal']:.1f} to {std['framework']:.1f} GB",
             reproduced=std["minimal"] <= p["std_hbm_gb"] <= std["framework"],
             note="lower: Algorithm 0/3 as written; upper: scale, mask, softmax, dropout as separate kernels"),
        dict(claim="FA1 Fig.2: FlashAttention HBM R/W", published=f"{p['fa_hbm_gb']} GB",
             model=f"{fa_floor:.1f} (compulsory) to {fa_alg[128]:.1f} (Alg. 1+4, Bc=128) GB",
             reproduced=fa_floor <= p["fa_hbm_gb"] <= fa_alg[128],
             note="Alg. 1+4 without cache reuse: " + ", ".join(f"Bc={k}: {v:.1f} GB" for k, v in fa_alg.items())),
        dict(claim="FA1 Fig.2: standard attention is HBM-bound", published=f"{p['std_ms']} ms",
             model=f">= {p['std_hbm_gb'] * 1e9 / (gpu.hbm_gbps * 1e9) * 1e3:.1f} ms from HBM alone",
             reproduced=p["std_hbm_gb"] / gpu.hbm_gbps * 1e3 <= p["std_ms"],
             note=f"measured runtime implies {p['std_hbm_gb'] / (p['std_ms'] / 1e3):.0f} GB/s, "
                  f"{100 * p['std_hbm_gb'] / (p['std_ms'] / 1e3) / gpu.hbm_gbps:.0f}% of the 1,555 GB/s peak"),
        dict(claim="FA1 Fig.2: FlashAttention is NOT HBM-bound", published=f"{p['fa_ms']} ms",
             model=f"HBM alone needs only {p['fa_hbm_gb'] / gpu.hbm_gbps * 1e3:.1f} ms",
             reproduced=p["fa_hbm_gb"] / gpu.hbm_gbps * 1e3 < 0.5 * p["fa_ms"],
             note="the remaining time is compute and non-matmul work, which FA2 later attacks"),
    ]
    # Theorem 2 scaling: FA traffic falls as 1/Bc (i.e. as 1/M) toward the compulsory floor.
    ratios = [fa_alg[64] / fa_alg[128], fa_alg[128] / fa_alg[256]]
    rows.append(dict(claim="FA1 Thm. 2 / Fig.2 middle: HBM accesses fall as block size grows",
                     published="monotone decrease, Theta(N^2 d^2 / M)",
                     model=f"x{ratios[0]:.2f} from Bc 64->128, x{ratios[1]:.2f} from 128->256",
                     reproduced=all(r > 1 for r in ratios),
                     note="halving passes over Q/O per doubling of Bc, approaching the 12-pass floor"))
    # FLOP count consistency check (flagged, not used)
    fwd_matmul = 4 * N * N * d * B * H / 1e9
    rows.append(dict(claim="FA1 Fig.2: GFLOPs column", published=f"{p['std_gflops']} / {p['fa_gflops']} GFLOPs",
                     model=f"{fwd_matmul:.0f} GFLOPs for the forward matmuls alone (4 N^2 d per head)",
                     reproduced=None,
                     note="not reconcilable from the text at the stated shape; not used in any check"))
    return rows


def check_fa2():
    rows = []
    gpu = A100_80GB
    b64, b128 = fa2_bounds(gpu, 64), fa2_bounds(gpu, 128)
    meas = FA2_FWD_D64[False]
    rows.append(dict(
        claim="FA2 Fig.5: forward TFLOPs/s, d=64, non-causal", published=f"{min(meas)} to {max(meas)} (fig. text layer)",
        model=f"{b64['serial_tflops']:.0f} (no overlap) to {b64['overlapped_tflops']:.0f} (perfect overlap)",
        reproduced=b64["serial_tflops"] <= min(meas) and max(meas) <= b64["overlapped_tflops"],
        note="bounds from tensor-core, FP32 and SFU work per score"))
    rows.append(dict(
        claim="FA2 Sec.4: up to 230 TFLOPs/s (73% of 312)", published=f"{FA2_HEADLINE_TFLOPS} ({FA2_HEADLINE_PCT}%)",
        model=f"d=128: {b128['serial_tflops']:.0f} to {b128['overlapped_tflops']:.0f}",
        reproduced=b128["serial_tflops"] <= FA2_HEADLINE_TFLOPS <= b128["overlapped_tflops"],
        note="head dim of the headline is not stated; d=128 assumed (FA2 benchmarks d=64 and d=128)"))
    rows.append(dict(
        claim="FA2: efficiency rises with head dimension", published="d=128 above d=64 in Fig. 5",
        model=f"no-overlap bound {b64['serial_tflops'] / 3.12:.0f}% (d=64) -> {b128['serial_tflops'] / 3.12:.0f}% (d=128)",
        reproduced=b128["serial_tflops"] > b64["serial_tflops"],
        note="non-matmul work per score is fixed while matmul work grows with d"))
    rows.append(dict(
        claim="FA2 Sec.3.1: a non-matmul FLOP costs 16x a matmul FLOP", published="16x (312 vs 19.5 TFLOPs/s)",
        model=f"{gpu.tensor_fp16_tflops / gpu.fp32_tflops:.1f}x from datasheet; per-SM rates give "
              f"{gpu.tensor_flops_per_clk / (gpu.fp32_lanes * 2):.0f}x",
        reproduced=round(gpu.tensor_fp16_tflops / gpu.fp32_tflops) == 16, note=""))
    # 3-10x over standard attention, forward, A100 80GB, tokens fixed at 16k as in FA2 Sec. 4.1
    lo, hi = [], []
    for n in (512, 1024, 2048, 4096, 8192, 16384):
        B, H, d = 16384 // n, 32, 64
        flops = 4 * n * n * d * B * H
        matmul_s = flops / (gpu.tensor_fp16_tflops * 1e12)
        for passes, acc in ((4, lo), (10.5, hi)):  # fwd N^2 passes: minimal vs framework graph
            std_s = passes * n * n * B * H * FP16 / (gpu.hbm_gbps * 1e9) + matmul_s
            fa2_s = flops / (min(meas) * 1e12)
            acc.append(std_s / fa2_s)
    rows.append(dict(
        claim="FA2 Sec.4: 3-10x faster than standard attention", published="3 to 10x",
        model=f"{min(lo):.1f}x to {max(hi):.1f}x (forward, seq 512-16k)",
        reproduced=FA2_VS_STANDARD[0] <= min(lo) and max(hi) <= FA2_VS_STANDARD[1],
        note="standard = HBM-bound N^2 passes + ideal matmul time; FA2 at its measured d=64 rate"))
    return rows


def main(out_dir="analysis"):
    os.makedirs(out_dir, exist_ok=True)
    rows = check_fa1_fig2() + check_fa2()
    with open(os.path.join(out_dir, "fa_reproduction.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim", "published", "model", "reproduced", "note"])
        w.writeheader()
        w.writerows(rows)
    mark = {True: "yes", False: "NO", None: "n/a"}
    lines = ["# Reproducing FlashAttention results without a GPU", "",
             "| Claim | Published | Model (no GPU) | Within bounds? | Note |", "|---|---|---|---|---|"]
    lines += [f"| {r['claim']} | {r['published']} | {r['model']} | {mark[r['reproduced']]} | {r['note']} |" for r in rows]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "fa_reproduction.md"), "w") as fh:
        fh.write(text)
    print(text)
    return rows


if __name__ == "__main__":
    main()
