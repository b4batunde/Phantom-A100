# A100 results for autotune-v1: what held, what did not (Measured)

Run on 2026-11-08 on one NVIDIA A100-SXM4-80GB (RunPod, CUDA 12.8). BH = 324, N = 128, d = 64, FP16:
the simulated BH = 42 scaled to 108 SMs. Median of 200 launches, CUDA events. All 23 design points passed
the correctness check. The frozen subject (`autotune-v1-sim-only`) and the pre-registration
(`autotune-v1-prereg`) were pushed to GitHub before the run. `python -m autotune.freeze verify` passed on the
GPU machine and again here.

**Nsight Compute could not read performance counters on this pod (`ERR_NVGPUCTRPERM`).** This is timing-only
validation. Mechanism fidelity is limited to registers per thread, and H7 and H9 are inconclusive.

Raw data and the scored report: `hw_measurements.json`, `scores.json`, `validation.md`, `validate.log`.
Re-scoring `hw_measurements.json` here with `python -m autotune.validate_hw --rescore` reproduces
`validation.md` and `scores.json` byte for byte.

## Pre-registered hypotheses (scored mechanically by `validate_hw.py`)

| ID | Question | Result | Value |
|---|---|---|---|
| H1 | Simulated cycles rank the 15 points like A100 time | **supported, at the threshold** | Spearman 0.800 (criterion >= 0.8); Kendall 0.657 |
| H2 | v4 `fused_tc` is the fastest of all 23 | **supported** | 0.069 ms |
| H3 | v2 tiled faster than v3 fused (simulator ordering) | **reversed** | v3/v2 time = 0.80 (simulated 1.65) |
| H4 | Simulated gains of register caps overstate hardware gains | **supported** | 6 of 6 pairs |
| H5 | Every step on the search path keeps its direction | **partial** | 3 of 4: the unlock step was predicted slower and is faster |
| H6 | The selected kernel is within 5% of the measured best | **supported** | regret 1.00x |
| H7 | Same-definition counters predict better than approximate/proxy ones | **inconclusive** | counters unobservable; registers 94% |
| H8 | Double buffering (rule only, no simulator evidence) is faster | **refuted** | 1 of 4 configs faster by > 2% |
| H9 | The frozen search on hardware counters also ends at `fused_tc` | **inconclusive** | counters unobservable |

## What this says

**The final decision held.** The optimizer's pick, v4, is the fastest of all 23 kernels on the A100, including
the double-buffered ones the simulator could not judge.

**The trust model held.** Every register-capped kernel did worse on hardware than the simulator predicted
(6 of 6). Five were slower than their uncapped versions; the sixth gained 1.09x where the simulator promised
1.93x. The static spill check was the right reason to reject them.

**The explanation of the search path did not hold.** The simulator put v3 1.65x slower than v2; on the A100,
v3 is 1.25x faster. This was named as a risk before the run: the simulator sizes v3 at 242 registers per
thread, the sm_80 build at 168. So the "unlock" move, which was presented as what carried the search past v2,
was compensating for a simulator artifact. On hardware, fusing directly from v1 is the better first step
(5.20x vs 4.17x for tiling). The search still reached v4, but not for the reason it reported.

**Magnitudes are not trustworthy.** Tensor cores: 3.46x measured vs 10.75x simulated. Vectorization: 1.41x vs
2.49x. Ranks survive better than ratios.

**The rule beat the simulator once.** For `grow_br` (128 query rows per block), the declared signature predicted
"faster" correctly in 90% of transitions, the simulator in 33%. The simulator ran the BH = 42 slice; the
full-chip result is consistent with the mechanism the rule names (each K/V tile reused by more rows), though
without counters that mechanism is not confirmed.

**The one prediction with no simulator evidence failed.** Double buffering with `cp.async` did not help in 3 of
4 configurations (H8).

## Post-hoc analysis (not pre-registered)

These were computed after seeing the data. They describe it; they were not predicted.

- **Structural vs tuning decisions.** Across the 51 simulated transitions, the simulator called the performance
  direction correctly for 18 of 19 *structural* transformations (tile, fuse, vectorize, tensor cores; the one
  miss is the v2/v3 reversal) and for 8 of 32 *tuning* transformations (tile shapes, register caps).
- **The tuning misses are mostly blindness, not wrong calls.** For 24 of the 32 tuning transitions the
  simulator predicted no change (within 2%); hardware showed a real difference in 22 of those. When the
  simulator did predict a direction for a tuning change, it was right in 6 of 8.
- **Inside the vectorized family** the simulator puts all 8 configurations within 1.2% of each other; on the
  A100 they span 2.2x (Spearman 0.55 within the family).

Working conclusion, to be tested rather than assumed: **in this setup, simulator counters were reliable for
structural decisions and for rejecting known blind spots, and unreliable for tile-level tuning and for absolute
speedups.** One kernel, one workload, one GPU.

## Known defect (recorded before the run)

`smem_limited_occupancy` can never fire (ADDENDUM-1). It did not affect any decision here.

## Open

- **Mechanism fidelity (H7) and the search on hardware counters (H9)** need Nsight counters. They need a GPU
  host that allows profiling (a full VM, not a container that blocks `ERR_NVGPUCTRPERM`).
- Nothing in autotune-v1 was changed. Hardware-informed changes are listed in `autotune/V2_BACKLOG.md`.
