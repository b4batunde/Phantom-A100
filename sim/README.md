# Running the kernels without a GPU (GPGPU-Sim)

The kernels in `csrc/kernels/` are plain CUDA (no PyTorch), so `sim/harness.cu` can drive them on
GPGPU-Sim, a cycle-level GPU simulator that runs on a normal Linux CPU. The same binary also runs
on a real GPU. Results from here are labeled **Simulated** (`results/simulated/`).

What it gives you without a GPU:
- **Correctness**: every run checks the kernel's output against a double-precision CPU reference.
- **Timing**: simulated cycles, warp instructions, occupancy, L2 and DRAM traffic on an A100 model.

## Setup that produced `results/simulated/` (Ubuntu 24.04, 2 cores, about 30 minutes)

```bash
apt-get install -y build-essential bison flex zlib1g-dev xutils-dev cmake nvidia-cuda-toolkit   # CUDA 12.0
git clone -b dev https://github.com/accel-sim/gpgpu-sim_distribution.git gpgpu-sim
cd gpgpu-sim && git checkout 4195dd4
git apply ../cuda-attention/sim/patches/all-fixes-for-gpgpu-sim-dev-4195dd4.patch

# one CUDA root with bin/, include/, lib64/
mkdir -p /opt/cuda12/bin
ln -sf /usr/lib/nvidia-cuda-toolkit/bin/* /opt/cuda12/bin/
for t in cuobjdump nvdisasm ptxas fatbinary nvlink; do ln -sf /usr/bin/$t /opt/cuda12/bin/$t; done
ln -sfn /usr/include /opt/cuda12/include && ln -sfn /usr/lib/x86_64-linux-gnu /opt/cuda12/lib64
export CUDA_INSTALL_PATH=/opt/cuda12 PATH=/opt/cuda12/bin:$PATH
cmake -B build && cmake --build build -j2 && cmake --install build
```

Use CUDA 12, not 13: CUDA 13's host code calls `__cudaLaunchKernel` / `__cudaGetKernel`, which the
simulator's runtime does not provide, and its versioned `libcudart.so.13` symbols do not resolve.

Build the harness as PTX for sm_75. The simulator does not implement `cp.async`; the sm_75 path of
`fused_db` falls back to synchronous copies (the A100 config still simulates an A100):

```bash
cd cuda-attention
nvcc -O3 -std=c++17 --cudart shared -gencode arch=compute_75,code=compute_75 -o sim/harness sim/harness.cu
```

## Run

`setup_environment` may pick a library directory named after a different gcc version than the one
that built the simulator. If the harness reports "no CUDA-capable device", set the paths yourself
(the PTX variables are what `setup_environment` exports for CUDA >= 6):

```bash
export LD_LIBRARY_PATH=/path/to/gpgpu-sim/lib/gcc-<your gcc>/cuda-12000/release:$LD_LIBRARY_PATH
export PTX_SIM_USE_PTX_FILE=1.ptx PTX_SIM_KERNELFILE=_1.ptx CUOBJDUMP_SIM_FILE=jj
```

```bash
source /path/to/gpgpu-sim/setup
# correctness: all 6 kernels, N=64 and N=100 (not a tile multiple), causal off/on  (~10 min)
python sim/run_sim.py --runs naive:2:64:0 ... --out results/simulated/correctness.csv
# performance: same workload for every kernel  (~2 h on 2 cores; naive dominates)
python sim/run_sim.py --runs naive:42:128:0 tiled:42:128:0 fused:42:128:0 fused_vec:42:128:0 fused_tc:42:128:0 \
    --out results/simulated/perf.csv
python sim/report.py --correctness results/simulated/correctness.csv --perf results/simulated/perf.csv
```

`configs/A100_slice` is the stock SM80_A100 config with 14 of 108 SMs and 5 of 40 memory channels.
That keeps per-SM DRAM bandwidth and L2 capacity close to the real chip at about 1/8 of the
simulation cost. `configs/A100_full` is the whole chip. BH=42, N=128 gives 84 blocks of 64 query
rows: exactly 1 wave for v3.1 (6 blocks/SM), 3 for v4 (2/SM) and 1.5 for v3 (4/SM).

Simulation speed is 100K to 500K thread-instructions per second, so large shapes are impractical;
naive at B=4, H=16, N=1024 would take most of a day.

## Simulator and config fixes (found while running these kernels)

| Fix | Symptom | Cause |
| --- | --- | --- |
| `gpgpusim-wmma-half-fix.patch` | segfault in `mma_impl` on any `wmma.mma` | FP16 operand bits were cast to a pointer and dereferenced |
| `gpgpusim-wmma-shared-space-fix.patch` | wrong results when fragments are loaded/stored in shared memory | the lexer folds `.shared` into the shape token and the grammar hard-codes the global space |
| `gpgpusim-wmma-fp32-accumulate-fix.patch` | ~1e-2 error with FP32 accumulators | the 16-term dot product was summed in FP16 |
| `gpgpusim-shfl-dual-destination-fix.patch` | every `__shfl_xor_sync` returned 0 or garbage; v2's softmax output 1.0 everywhere | nvcc emits `shfl.sync d\|p`; the parser tags any non-`setp` `a\|b` destination as (predicate, data), so the value went into `p` and a flag into `d` |
| SHFL latency in `configs/*/gpgpusim.config` | ~22,000 cycles per shuffle; v2's softmax took 445K cycles for 32 rows | the stock SM80_A100 config lists 5 integer latencies but the simulator reads 6; SHFL (the 6th) was uninitialized. Set to 21 cycles / initiation 4, the SM7_QV100 values |

`all-fixes-for-gpgpu-sim-dev-4195dd4.patch` combines the four code patches for the commit above (the
shared-space patch needed rebasing: this commit has no `vp_spec` rule). The individual patches are
kept for older commits.

After the fixes, a standalone shuffle test and a standalone WMMA test are exact, and all 24 kernel
correctness runs pass (`results/simulated/correctness.csv`).

## Limits

- GPGPU-Sim executes **PTX**, not the SASS a real GPU runs: no ptxas scheduling or register-level
  optimization. Treat cycle counts as estimates; ratios between versions are more reliable than
  absolute numbers.
- `cp.async` is not implemented, so v3.2's double buffering cannot be evaluated here.
- FlashAttention-2 and PyTorch SDPA cannot run here, so % of FlashAttention-2 needs a Measured run.
