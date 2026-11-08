# autotune: counter-guided kernel optimization without the target GPU

**Question:** can architectural simulator counters drive an interpretable optimization policy that
selects useful CUDA kernel transformations, without access to the target GPU?

In the original Phantom A100 study a person read GPGPU-Sim's counters and decided what to change
next (v1 -> v2 -> v3 -> v3.1 -> v4). This package makes that loop executable:

```
design point -> GPGPU-Sim -> counters -> bottleneck rules -> ranked transformations
     ^                                                              |
     |        judge: predicted vs observed counters, update beliefs |
     +-------------------- best trusted candidate <-----------------+
```

Results: [`results/autotune/README.md`](../results/autotune/README.md) (Simulated) and, after a GPU
session, `results/autotune/measured/validation.md` (Measured).

## Pieces

| File | What it holds |
|---|---|
| `space.py` | The design space: 23 runnable kernel configurations (structure, tiles, FP16 vectorization, tensor cores, register cap, double buffering), 15 of which the simulator can judge. The historical v1 to v4 kernels are 6 of them. |
| `counters.py` | Parses a simulator log into raw counters and workload-normalized metrics (L2 traffic vs compulsory, instructions per FLOP, scoreboard stall share, occupancy limiter, waves). |
| `diagnose.py` | Nine explicit bottleneck rules with thresholds and printable evidence. No learned model. |
| `transforms.py` | Eleven transformations. Each declares the bottlenecks it targets and its **expected counter signature** (direction of each counter) before anything runs. Register caps, tile shapes and double buffering are real options that often do not help (the decoys). |
| `policy.py` | Predict-then-verify: compares declared vs observed counter changes, gives a verdict (`confirmed`, `mechanism_no_gain`, `gain_unexplained`, `refuted`), and keeps a Beta confidence per (bottleneck, transformation) plus the observed sign of every counter. |
| `search.py` | Counter-guided search, plus two baselines under the same simulation budget: unguided local search (same menu, no counters) and random search. |
| `backends.py` | GPGPU-Sim runner with a deterministic cache, a replay backend for fair strategy comparisons, and static `ptxas -v` facts for sm_80. |
| `groundtruth.py` | Simulates every sim-verifiable point once. |
| `experiments.py` | Rediscovery from v1, robustness from every start, same-budget comparison, signature audit, greedy ablation. |
| `freeze.py` | Writes and verifies the pre-registration: manifest (frozen commit, SHA-256 of every frozen file) and machine-readable predictions with hypotheses H1-H9. |
| `hw_counters.py` | The explicit simulator-counter -> Nsight Compute metric mapping (same_definition / approximate / proxy), the optional ncu runner and parser. Missing values are UNOBSERVABLE. |
| `validate_hw.py` | On a real GPU: times all 23 points, optionally profiles them, and scores the frozen predictions as performance, mechanism and decision fidelity. `--rescore` re-scores saved measurements without a GPU. |
| `run.py` | One live search on the simulator for a new workload or starting point. |

## Two mechanisms worth calling out

**Trust.** A simulated speedup is only accepted if nothing known makes it unreliable. GPGPU-Sim
executes PTX, which has no register spills, so a register cap looks free in simulation. The
optimizer compiles every kernel for sm_80 with `ptxas -v` and marks any spilling point as
untrusted: it is reported, it teaches the signature model, but it never becomes the incumbent.
Double buffering (`cp.async`) is not implemented in the simulator at all, so those points are
listed as hardware-only candidates and never simulated.

**Unlock moves.** Greedy search stops at v2: at N = 128, fusion alone is slower than tiling
(v3 is register-limited). The guided search allows one non-improving move, and only when the
counters justify it: the incumbent has a diagnosed bottleneck that no transformation can fix from
its current structure (CUDA-core math, whose remedy, tensor cores, needs a fused kernel), and the
move makes that remedy applicable. The ablation in `experiments.py` runs the search without it.

## Reproduce

```bash
# simulator: see sim/README.md (about 30 minutes to build on 2 cores)
nvcc -O3 -std=c++17 --cudart shared -gencode arch=compute_75,code=compute_75 -o sim/harness sim/harness.cu
python -m autotune.groundtruth --jobs 2          # about 1 hour on 2 cores for BH=42, N=128
python -m autotune.experiments                   # seconds, from the cache
python -m pytest -q tests/test_autotune.py       # CPU-only unit tests

# pre-registration (done once, before any hardware data; tag autotune-v1-prereg)
python -m autotune.freeze write && python -m autotune.freeze verify

# on a rented GPU (also part of scripts/run_all.sh); --profile off for timing only
python -m autotune.validate_hw
```

## Limits (stated up front)

- The menu, thresholds and signatures were written by someone who knew the v1 to v4 history.
  The decoys and the robustness runs reduce but do not remove that bias. The real test is a
  kernel the menu was not written for (GEMM, reduction, normalization): next step.
- One workload so far. Thresholds are first-principles values, not fitted, but they are untested
  on other shapes.
- The design space is small (15 simulated points), so "simulations to reach the best kernel" is
  the meaningful comparison, not whether search finds it eventually.
- The simulator and the real sm_80 build disagree on v3's registers: GPGPU-Sim's occupancy model
  uses 242 registers per thread (ptxas on the compute_75 PTX the simulator runs), while the native
  sm_80 build uses 168. The "v3 is register-limited, so fusion alone loses at N = 128" story, and
  with it the need for the unlock move, may not hold on hardware. `validate_hw.py` will show it.
- Everything in `results/autotune/README.md` is Simulated. Whether the decisions hold on an A100
  is exactly what `validate_hw.py` measures.
