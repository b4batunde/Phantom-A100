#!/usr/bin/env bash
# Nsight Compute profile of every kernel at N = 4096 (skips the first call: v1/v2 launch 3 kernels per call, v3 launches 1).
# Outputs results/measured/ncu_<impl>.ncu-rep (open in the Nsight Compute GUI) and a CSV of key metrics.
#
#   bash bench/profile.sh            # N = 4096
#   N=8192 bash bench/profile.sh
set -euo pipefail
N="${N:-4096}"
mkdir -p results/measured

# Metrics that explain each version's speed:
#   dram__throughput                      how hard HBM is being pushed
#   sm__throughput                        how busy the SMs are
#   sm__warps_active (achieved occupancy) latency hiding
#   l1tex__data_bank_conflicts_pipe_lsu_mem_shared  shared-memory bank conflicts
#   dram__bytes_read / write              total HBM traffic (the N^2 term shows up here)
METRICS="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,dram__bytes_read.sum,dram__bytes_write.sum"

for impl in naive tiled fused fused_vec fused_db fused_tc; do
  echo "== $impl (N=$N)"
  ncu --set full --launch-skip 3 --launch-count 3 \
      -o "results/measured/ncu_${impl}_N${N}" -f \
      python bench/run_one.py --impl "$impl" --n "$N"
  ncu --metrics "$METRICS" --launch-skip 3 --launch-count 3 --csv \
      python bench/run_one.py --impl "$impl" --n "$N" > "results/measured/ncu_${impl}_N${N}_metrics.csv"
done

# Timeline view (kernel launches, gaps between the 3 kernels of v1/v2).
nsys profile -o "results/measured/nsys_all_N${N}" -f true \
    python bench/benchmark.py --ns "$N" --impls naive tiled fused fused_vec fused_db fused_tc --causal off --iters 5 --warmup 2 \
    --out "results/measured/nsys_bench_N${N}.csv"
echo "done: results/measured/ncu_*_N${N}.ncu-rep"
