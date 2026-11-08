# Addendum 2 (written 2026-11-08, after the timing-only A100 run, before the Nsight run)

## Why

The timing-only run showed that one "direction accuracy" number hides two different failure modes. For
tile-shape and register-cap changes, the simulator predicted a tie (within 2%) in 24 of 32 transitions, and
the A100 resolved 22 of those 24 into real differences. When the simulator did commit to a direction, it was
right in 6 of 8. The simulator was mostly **insensitive** there, not mostly wrong. The declared signatures show
the opposite pattern: they always commit, and for tuning they were opposite to hardware in 43% of claims.

## What is added

From now on every direction analysis reports, separately, for the simulator and for the declared signatures,
per counter, per mapping equivalence and per transformation family:

| Measure | Definition |
|---|---|
| Coverage | committed predictions (up/down, faster/slower) / all claims including ties |
| Conditional accuracy | committed predictions in the hardware direction / committed predictions |
| Confidently wrong | committed predictions opposite to hardware / committed predictions |
| Tie resolution | predicted ties that hardware resolved into a real difference / predicted ties |
| Overall accuracy | matches / all claims (the pre-registered metric, unchanged) |

Claims exclude `?` (no prediction) and unobservable hardware values. Tolerances are unchanged: 3% for counters,
2% for performance.

Transformation families, fixed now:

- **structural:** `tile_smem`, `fuse`, `vectorize_fp16`, `tensor_cores`
- **tuning:** `reg_cap_128`, `remove_reg_cap`, `grow_br`, `shrink_br`, `grow_bc`, `shrink_bc`
- **rule_only:** `double_buffer`

Implementation: `autotune/decompose.py`, writing `decomposition.{json,md}` next to `scores.json`.
`validate_hw.py` calls it after scoring.

## Status of each use

- **Timing-only run (already measured):** the decomposition is post-hoc there, because it was defined after
  seeing the data. It is reported as such.
- **Nsight run (H7, H9; not yet measured):** the decomposition is specified here in advance and applies to
  every counter. No new pass/fail hypothesis is added; H1 to H9 and their criteria are unchanged.

Nothing in the frozen optimizer changes. `python -m autotune.freeze verify` passes.
