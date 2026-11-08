# Addendum 1 (written 2026-11-08, before any hardware measurement)

Found while writing the Nsight Compute instrumentation, after the freeze and the pre-registration
commit, and before any A100 data exists. Recorded here instead of fixed, because autotune-v1 is
frozen.

## Defect: `smem_limited_occupancy` can never fire

`autotune/diagnose.py` checks `"smem" in cta_limiter`, but GPGPU-Sim reports the shared-memory
limiter as `shmem` ("CTA/core = 6, limited by: shmem"), and `"smem"` is not a substring of
`"shmem"`. The hardware adapter uses the simulator's vocabulary (`shmem`), so the rule cannot fire on
hardware counters either.

Effect on the simulated results: none on the search path. The only point the simulator reports as
shared-memory limited is `fused/br64/bc32/r128` (occupancy 18.6%), and the only transformation that
targets this bottleneck, `shrink_bc`, applies to `fused_vec` points, not `fused`. `shrink_bc` was
therefore never proposed, and it would not have been proposed from any simulated point.

Status: left as is in autotune-v1 and scored as is. Fix belongs in autotune-v2.

## Clarification: the frozen trust check on hardware data

H9 reruns the frozen policy on hardware counters. Its trust check still rejects points that spill
on sm_80 (the check exists because the simulator cannot see spills, while hardware time does include
them) and still never evaluates double-buffered points. This is the frozen behaviour, not a
hardware-informed choice. H8 and the per-transition tables measure the double-buffered points
separately.
