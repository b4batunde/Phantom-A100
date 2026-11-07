#!/usr/bin/env bash
# One paid-GPU session, start to finish (the plan's "scripted to finish in one session").
#
#   bash scripts/run_all.sh            # on a fresh A100 / L4 / H100 box with PyTorch + CUDA
#
# Steps: environment record -> build -> flash-attn -> tests (gate) -> benchmark -> tile sweep
#        -> plots + summary -> Nsight profiles (if ncu exists) -> results.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/measured
log() { printf '\n==== %s ====\n' "$*"; }

log "environment"
{ nvidia-smi; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"; nvcc --version || true; } \
  | tee results/measured/environment.txt

log "build extension"
pip install -q ninja pytest matplotlib
MAX_JOBS="${MAX_JOBS:-4}" pip install -q -e . --no-build-isolation

log "flash-attn baseline"
if python -c "import flash_attn" 2>/dev/null; then echo "already installed"
elif python -c "import torch,sys;sys.exit(torch.cuda.get_device_capability()[0] < 8)"; then
  pip install -q flash-attn --no-build-isolation || echo "flash-attn install failed: benchmark will mark flash2 unavailable"
else echo "pre-Ampere GPU: no FlashAttention-2 baseline"; fi

log "correctness gate"
python -m pytest -q tests | tee results/measured/tests.txt

log "benchmark sweep"
python bench/benchmark.py
CSV=$(ls -t results/measured/bench_*.csv | head -1)

log "tile sweep"
python bench/sweep_tiles.py
python bench/sweep_tiles.py --causal

log "plots + headline"
python bench/plot.py "$CSV"

if command -v ncu >/dev/null; then
  log "profiling"
  bash bench/profile.sh || echo "profiling failed (often a permissions setting on rented GPUs); results kept"
else
  echo "ncu not found: skipping Nsight profiles"
fi

tar czf results_measured.tar.gz results/measured
log "done -> results_measured.tar.gz (download it before the instance shuts down)"
