# autotune-v2 backlog

Changes motivated by the A100 run (`results/autotune/measured/FINDINGS.md`). None of these are applied to
autotune-v1, which stays frozen as the experimental subject. Each v2 change should be reported against the v1
results, and anything v2 claims needs its own pre-registration and a fresh hardware run.

| # | Observation (Measured) | Candidate change for v2 |
|---|---|---|
| 1 | Simulator sizes v3 at 242 registers, sm_80 at 168; v2/v3 order reversed on hardware (H3) | Compute occupancy from the target's native `ptxas` registers instead of the simulator's compute_75 PTX count; flag points where the two disagree as partially trusted |
| 2 | The unlock move was only needed because of item 1 | Re-test greedy vs unlock after fixing item 1, before keeping the unlock move |
| 3 | Simulator predicted a tie for 24 of 32 tuning transitions; hardware resolved 22 of them; committed predictions were right 6 of 8 (coverage 25%, conditional accuracy 75%) | Treat a simulated tie as *insufficient information*, not as evidence of no effect (see item 10) |
| 4 | `grow_br`: declared rule 90% right, simulator 33% | Track rule accuracy and simulator accuracy separately per transformation; let the confidence model prefer whichever has been right |
| 5 | Magnitudes overstated (tensor cores 10.75x sim vs 3.46x measured) | Use ranks and directions only; drop magnitude-based scoring |
| 6 | Double buffering failed with no simulator evidence (H8) | Rule-only predictions start at low confidence |
| 7 | `smem_limited_occupancy` never fires (`"smem"` vs `"shmem"`, ADDENDUM-1) | Fix the limiter string match |
| 8 | Slice workload (BH = 42, 14 SMs) missed full-chip tile effects | Simulate a second, larger slice or wave count before judging tile shapes |
| 9 | Nsight blocked on RunPod containers | Rerun validation on a host with counter access to score H7 and H9 for v1 before any v2 claims |
| 10 | Simulation is informative when it commits and insensitive when it predicts a tie; rules commit everywhere and are often confidently wrong on tuning | **Selective optimizer:** decide from simulation only when it carries enough signal (a committed, separated prediction); otherwise abstain and escalate the candidates to hardware measurement. Evaluate by coverage, conditional accuracy and hardware measurements saved, not by overall accuracy alone. Wait for the H7/H9 counter-level decomposition before designing it |
