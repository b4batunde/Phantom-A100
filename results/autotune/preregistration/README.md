# Pre-registration: autotune-v1 (written before any A100 measurement)

- Frozen commit: `a86168113af84d98002714061e399231dc0ce3c9` (tag `autotune-v1-sim-only`), committed 2026-11-07T18:28:10-07:00
- Written: 2026-11-08T02:41:27+00:00
- Manifest hash: `c011a933ce15f33f169206648a4ec65f953b06392f6ae44d9ebace18949379cd`
- Files: `manifest.json` (SHA-256 of every frozen file), `predictions.json` (everything below, machine-readable)

All optimization rules, counter signatures, confidence logic and the search algorithm were specified, and all predictions below were written, before any A100 measurement was observed. Nothing in `files` may change; hardware-informed changes go to autotune-v2.

Check the tree still matches: `python -m autotune.freeze verify`. `validate_hw.py` runs the same check and prints the result at the top of the hardware report.

## Hypotheses

**H1 (performance fidelity).** Simulated cycles rank the 15 sim-verifiable points in nearly the same order as measured A100 time.
  Metric: spearman_rho(sim_cycles, measured_ms) over the 15 sim-verifiable points. Criterion: supported: >= 0.8; partial: >= 0.5 and < 0.8; refuted: < 0.5

**H2 (performance fidelity).** v4 fused_tc, the simulator's best trusted point, is the fastest of all 23 points on the A100.
  Metric: argmin(measured_ms) over all 23 points == 'fused_tc'. Criterion: supported: true; refuted: false

**H3 (performance fidelity (v2/v3 ordering)).** Simulator ordering: v2 tiled is faster than v3 fused (simulated v3/v2 time ratio 1.65).
  Risk identified in advance: The simulator's occupancy model gives v3 242 registers/thread (ptxas on the compute_75 PTX it executes: 4 blocks/SM), while the native sm_80 build uses 168 (6 blocks/SM). If register-limited occupancy is what makes v3 slow in simulation, v3 may be close to or faster than v2 on the A100. If so, the unlock move that carried the search from v2 to v3 was needed only because of a simulator artifact, and pure greedy search on hardware counters would not have stopped at v2.
  Metric: measured_ms(fused/br64/bc32) / measured_ms(tiled). Criterion: simulator ordering holds: > 1.02; tie: 0.98 to 1.02; reversed: < 0.98

**H4 (trust model (register spills)).** Simulated gains of register-capped (spilling) points overstate hardware gains: that is why they were marked untrusted.
  Metric: for each of the 6 capped/uncapped pairs: measured speedup of capping < simulated speedup of capping minus 0.02. Criterion: supported: >= 4 of 6 pairs; refuted: <= 2 of 6 pairs; inconclusive: 3 of 6

**H5 (decision fidelity).** Every accepted step on the guided search path is in the same performance direction on the A100 (improving steps faster, the unlock step slower).
  Metric: per step on the path: perf_dir(measured parent/child) == predicted_performance. Criterion: supported: all steps match; partial: all improving steps match; refuted: an improving step is slower on hardware

**H6 (decision fidelity).** The kernel the frozen optimizer selected (fused_tc) is within 5% of the measured-fastest point among all 23.
  Metric: measured_ms(fused_tc) / min(measured_ms) <= 1.05. Criterion: supported: true; refuted: false

**H7 (mechanism fidelity).** Counters whose hardware metric has the same definition as the simulator counter predict direction better than counters mapped to an approximate or proxy metric.
  Metric: direction accuracy of simulator-predicted counter changes, grouped by mapping equivalence (same_definition / approximate / proxy), over all sim-verifiable transitions. Criterion: supported: accuracy(same_definition) > accuracy(approximate) and > accuracy(proxy); refuted: otherwise

**H8 (rule-only prediction (no simulator evidence)).** double_buffer (cp.async) makes fused_vec faster on the A100, as its declared signature says. The simulator could not test this.
  Metric: for each of the 4 tile configs without a register cap: measured speedup of db=1 over db=0. Criterion: supported: >= 3 of 4 faster by > 2%; refuted: <= 1 of 4 faster by > 2%; inconclusive: 2 of 4

**H9 (decision fidelity (search on hardware counters)).** Run unchanged on hardware counters (Nsight Compute mapped through the frozen mapping) and hardware time, the frozen policy also ends at fused_tc.
  Metric: final incumbent of counter_guided(start=naive) with HardwareBackend. Criterion: supported: ends at fused_tc; refuted: ends elsewhere; inconclusive: Nsight counters unavailable

## Simulated ranking being tested

| Rank | Point | Simulated cycles | Trusted |
|---|---|---|---|
| 1 | fused_tc (v4 fused_tc) | 52,659 | yes |
| 2 | fused_vec/br128/bc64/r128 | 559,186 | no |
| 3 | fused_vec/br128/bc64 | 559,326 | yes |
| 4 | fused_vec/br128/bc32/r128 | 561,541 | no |
| 5 | fused_vec/br128/bc32 | 561,725 | yes |
| 6 | fused_vec/br64/bc64/r128 | 563,691 | no |
| 7 | fused_vec/br64/bc64 | 563,975 | yes |
| 8 | fused_vec/br64/bc32/r128 | 565,540 | no |
| 9 | fused_vec/br64/bc32 (v3.1 fused_vec) | 565,972 | yes |
| 10 | fused/br128/bc32/r128 | 686,124 | no |
| 11 | fused/br64/bc32/r128 | 732,923 | no |
| 12 | tiled (v2 tiled) | 855,959 | yes |
| 13 | fused/br128/bc32 | 1,333,912 | yes |
| 14 | fused/br64/bc32 (v3 fused) | 1,410,914 | yes |
| 15 | naive (v1 naive) | 3,049,245 | yes |

Search path: naive -> tiled -> fused/br64/bc32 -> fused_vec/br64/bc32 -> fused_tc. Transitions with predictions: 51 simulated, 32 rule-only.

## Known simulator limitations (stated before measurement)

- Executes PTX, not SASS: no ptxas scheduling, no register spills. Register-capped points spill 168-188 bytes/thread on sm_80 at no simulated cost.
- cp.async is not implemented; double-buffered points run a synchronous fallback and are not simulated.
- Register counts for occupancy come from ptxas on the compute_75 PTX the simulator runs; they can differ from the native sm_80 build (v3: 242 simulated vs 168 sm_80).
- 14-SM slice with 5 of 40 memory channels: L2 capacity is 1/8 of the real chip, so absolute L2/DRAM traffic and anything that depends on L2 hits can differ on the full A100.
- Cycle counts are not calibrated against hardware; ratios between versions are the intended output.
- naive and tiled counters come from the original v1-v4 study's logs (same kernel code).
