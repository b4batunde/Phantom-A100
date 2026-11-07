"""Run one kernel a few times on one shape; the target for Nsight Compute / Nsight Systems.

    python bench/run_one.py --impl fused --n 4096
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cuda_attn  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--impl", choices=list(cuda_attn.KERNELS), required=True)
p.add_argument("--n", type=int, default=4096)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--heads", type=int, default=16)
p.add_argument("--causal", action="store_true")
p.add_argument("--reps", type=int, default=5)
a = p.parse_args()

q, k, v = (torch.randn(a.batch, a.heads, a.n, 64, device="cuda", dtype=torch.float16) for _ in range(3))
fn = cuda_attn.KERNELS[a.impl]
for _ in range(a.reps):
    fn(q, k, v, a.causal)
torch.cuda.synchronize()
print(f"ran {a.impl} N={a.n} x{a.reps}")
