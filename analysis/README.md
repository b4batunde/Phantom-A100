# Analysis (not results)

Pen-and-paper models that help explain the numbers. Nothing here is a result: project results are
either **Simulated** (`results/simulated/`, GPGPU-Sim) or **Measured** (`results/measured/`, a real GPU).

- `gpufree/model.py`: per-kernel operation counts and an ideal roofline bound from A100 datasheet rates.
- `gpufree/reproduce_fa.py` -> `fa_reproduction.md`: checks the FlashAttention papers' published claims
  against datasheet-based bounds (memory passes, FLOP rates).
- `gpufree/predict.py`: combines the two into full-chip predictions. Its earlier outputs were removed:
  they used v4 simulator timings taken while GPGPU-Sim's SHFL latency was uninitialized
  (see `sim/README.md`). Re-run it on `results/simulated/perf.csv` if you want predictions again.
