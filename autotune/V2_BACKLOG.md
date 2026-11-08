# autotune-v2 backlog

Changes motivated by the A100 run (`results/autotune/measured/FINDINGS.md`). None of these are applied to
autotune-v1, which stays frozen as the experimental subject. Each v2 change should be reported against the v1
results, and anything v2 claims needs its own pre-registration and a fresh hardware run.

| # | Observation (Measured) | Candidate change for v2 |
|---|---|---|
| 1 | Simulator sizes v3 at 242 registers, sm_80 at 168; v2/v3 order reversed on hardware (H3) | Compute occupancy from the target's native `ptxas` registers instead of the simulator's compute_75 PTX count; flag points where the two disagree as partially trusted |
| 2 | The unlock move was only needed because of item 1 | Re-test greedy vs unlock after fixing item 1, before keeping the unlock move |
| 3 | Simulator predicted "no change" for 24 of 32 tuning transitions; hardware differed in 22 | Treat "simulator says same" as *unverifiable*, not as evidence of no effect; leave tuning decisions to hardware or to a cheaper measured sweep |
| 4 | `grow_br`: declared rule 90% right, simulator 33% | Track rule accuracy and simulator accuracy separately per transformation; let the confidence model prefer whichever has been right |
| 5 | Magnitudes overstated (tensor cores 10.75x sim vs 3.46x measured) | Use ranks and directions only; drop magnitude-based scoring |
| 6 | Double buffering failed with no simulator evidence (H8) | Rule-only predictions start at low confidence |
| 7 | `smem_limited_occupancy` never fires (`"smem"` vs `"shmem"`, ADDENDUM-1) | Fix the limiter string match |
| 8 | Slice workload (BH = 42, 14 SMs) missed full-chip tile effects | Simulate a second, larger slice or wave count before judging tile shapes |
| 9 | Nsight blocked on RunPod containers | Rerun validation on a host with counter access to score H7 and H9 for v1 before any v2 claims |
