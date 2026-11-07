"""Analytical performance model for the attention kernels (no GPU needed).

Three ingredients:

1. GPU SPECS: datasheet numbers (sources in SPEC_SOURCES) plus per-SM pipe rates taken from
   the GPGPU-Sim SM80_A100 configuration, which are consistent with the datasheet totals
   (e.g. 108 SMs x 1.41 GHz x 64 FP32 lanes x 2 = 19.5 TFLOPS).

2. WORK PER SCORE: for every kernel, the operations needed per (query row, key) pair, counted
   by reading the kernel source (see the comments next to each entry). Units:
     fp32      thread-level FP32 / FP16-convert instructions on the FMA pipe
     sfu       exp2/exp evaluations on the special-function unit
     tensor    FLOPs on tensor cores
     smem_wf   shared-memory wavefronts (one 128-byte bank access) per score, assuming no bank conflicts
     lsu       thread-level global load/store instructions that hit in L1/L2 (32 per clock per SM)
   plus DRAM bytes per score for the N x N buffers (naive/tiled only).

3. ROOFLINE OVER ALL RESOURCES: time = max over resources of (work / rate). This is an IDEAL
   bound: perfect overlap, no latency stalls, no bank conflicts. The GPU simulator measures how
   far each kernel falls short of it (gpufree/predict.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- GPU specs
SPEC_SOURCES = {
    "a100_whitepaper": "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf",
    "a100_product": "https://www.nvidia.com/en-us/data-center/a100/",
    "cuda_cc_table": "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html",
    "gpgpusim_sm80": "gpgpu-sim_distribution/configs/tested-cfgs/SM80_A100/gpgpusim.config",
}


@dataclass(frozen=True)
class GPU:
    name: str
    sms: int
    clock_ghz: float
    hbm_gbps: float               # datasheet memory bandwidth
    fp32_tflops: float            # datasheet
    tensor_fp16_tflops: float     # datasheet, dense (no sparsity)
    # per-SM limits (CUDA guide, compute capability 8.0)
    regs_per_sm: int = 65536
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    smem_per_sm_kb: int = 164
    smem_reserved_per_block_kb: int = 1   # CUDA occupancy calculator convention for CC 8.x (assumption)
    # per-SM pipe rates per clock (GPGPU-Sim SM80_A100: 4 units each; fp initiation 2, sfu 8, tensor 16)
    fp32_lanes: int = 64                  # 4 units x 32 lanes / initiation 2
    sfu_lanes: int = 16                   # 4 units x 32 lanes / initiation 8
    tensor_flops_per_clk: int = 2048      # 4 units x (16x16x16 x 2 FLOPs) / initiation 16
    smem_wavefronts_per_clk: int = 1      # 32 banks x 4 B = 128 B per clock
    lsu_lanes: int = 32                   # L1 data path: 128 B per clock
    issue_per_clk: int = 4                # 4 warp schedulers, 1 instruction each
    sources: tuple = field(default=("a100_whitepaper", "a100_product", "cuda_cc_table", "gpgpusim_sm80"))

    def check(self):
        """The per-SM rates must reproduce the datasheet totals."""
        fp32 = self.sms * self.clock_ghz * 1e9 * self.fp32_lanes * 2 / 1e12
        tc = self.sms * self.clock_ghz * 1e9 * self.tensor_flops_per_clk / 1e12
        return {"fp32_tflops_from_rates": round(fp32, 1), "fp32_tflops_datasheet": self.fp32_tflops,
                "tensor_tflops_from_rates": round(tc, 1), "tensor_tflops_datasheet": self.tensor_fp16_tflops}


# A100 40GB SXM: whitepaper (108 SMs, 1410 MHz boost, 1555 GB/s, 19.5 / 312 TFLOPS dense).
A100_40GB = GPU("A100-SXM4-40GB", sms=108, clock_ghz=1.41, hbm_gbps=1555, fp32_tflops=19.5, tensor_fp16_tflops=312)
# A100 80GB SXM: product page (2,039 GB/s; 19.5 / 312 TFLOPS dense). Same GA100 SM configuration:
# 312 TFLOPS = 108 SMs x 1.41 GHz x 2048 FLOP/clk, so SM count and clock are inferred from the totals.
# This is the GPU the FlashAttention-2 paper benchmarks on.
A100_80GB = GPU("A100-SXM4-80GB", sms=108, clock_ghz=1.41, hbm_gbps=2039, fp32_tflops=19.5, tensor_fp16_tflops=312)
GPUS = {"a100_40gb": A100_40GB, "a100_80gb": A100_80GB}


# --------------------------------------------------------------------------- occupancy
@dataclass(frozen=True)
class LaunchShape:
    threads_per_block: int
    regs_per_thread: int          # from `ptxas -v` (sm_80, CUDA 13.2) -- see README
    smem_per_block_bytes: int
    rows_per_block: int           # query rows a block owns (fused kernels)


def occupancy(gpu: GPU, ls: LaunchShape) -> dict:
    warps = math.ceil(ls.threads_per_block / 32)
    regs_per_warp = math.ceil(ls.regs_per_thread * 32 / 256) * 256          # 256-register allocation unit
    by_regs = gpu.regs_per_sm // (regs_per_warp * warps)
    by_threads = gpu.max_threads_per_sm // ls.threads_per_block
    smem_kb = ls.smem_per_block_bytes / 1024 + gpu.smem_reserved_per_block_kb
    by_smem = int(gpu.smem_per_sm_kb // smem_kb) if ls.smem_per_block_bytes else gpu.max_blocks_per_sm
    blocks = max(0, min(by_regs, by_threads, by_smem, gpu.max_blocks_per_sm))
    limiter = min((("registers", by_regs), ("threads", by_threads), ("shared memory", by_smem),
                   ("block slots", gpu.max_blocks_per_sm)), key=lambda kv: kv[1])[0]
    return {"blocks_per_sm": blocks, "warps_per_sm": blocks * warps,
            "occupancy": blocks * warps * 32 / gpu.max_threads_per_sm, "limited_by": limiter}


# --------------------------------------------------------------------------- work per score
@dataclass(frozen=True)
class KernelCost:
    name: str
    label: str
    fp32: float
    sfu: float
    tensor: float
    smem_wf: float
    lsu: float
    dram_bytes_per_score: float   # N x N buffer traffic (0 for fused kernels)
    computes_masked: bool         # True: computes all N^2 scores even when causal
    launch: LaunchShape | None
    notes: str


D = 64
COSTS = {
    "naive": KernelCost(
        "naive", "v1 naive",
        fp32=D + 3 + D,        # QK dot (64 FFMA) + softmax max/sub/normalize + PV (64 FFMA per output column)
        sfu=2,                 # expf in the sum pass and again in the normalize pass
        tensor=0, smem_wf=0,
        lsu=2 * D + 4 + 2 * D,  # scores: q[c], k[c] per FMA; softmax 4 row accesses; PV: P and V per FMA
        dram_bytes_per_score=4 * 6,  # S write, softmax 3 reads + 1 write, PV read (fp32), assuming no reuse
        computes_masked=True,
        launch=LaunchShape(256, 32, 0, 16),
        notes="every operand is re-read from global memory; L1/L2 absorb most of it"),
    "tiled": KernelCost(
        "tiled", "v2 tiled",
        fp32=D + 4 + D,
        sfu=2,
        tensor=0,
        smem_wf=2 * D / 32 * 2,  # 2 shared loads per FMA (QK and PV), 1 wavefront per warp instruction
        lsu=4 / 32 * 32 + 2,     # tile loads amortized over 32 reuses, plus softmax row accesses
        dram_bytes_per_score=4 * 4,  # S write, softmax read + read + write, PV read
        computes_masked=True,    # scores yes; PV skips upper tiles (ignored here)
        launch=LaunchShape(1024, 32, 2 * 32 * 33 * 4, 32),
        notes="shared-memory bound: one output per thread means two shared loads per FMA"),
    "fused": KernelCost(
        "fused", "v3 fused",
        fp32=2 * D + 6,        # QK + PV FMAs, plus max / mask / rescale / row-sum bookkeeping
        sfu=1,
        tensor=0,
        smem_wf=(D / 4 * 2 + 2) / 32,  # per key: 16 LDS.128 of K + 16 of V (broadcast), S write + read; per warp of 32 rows
        lsu=0, dram_bytes_per_score=0, computes_masked=False,
        launch=LaunchShape(64, 236, 24832, 64),
        notes="FP32 FMA bound; one thread per row keeps q and o in 128 registers"),
    "fused_vec": KernelCost(
        "fused_vec", "v3.1 + 128-bit loads",
        fp32=2 * D + 2 * D + 6,  # same FMAs + one FP16->FP32 conversion per K and V element read from smem
        sfu=1, tensor=0,
        smem_wf=(D / 8 * 2 + 2) / 32,  # FP16 tiles: half as many LDS.128
        lsu=0, dram_bytes_per_score=0, computes_masked=False,
        launch=LaunchShape(64, 168, 2 * 32 * 64 * 2 + 64 * 33 * 4, 64),
        notes="FP16 shared tiles halve shared traffic but add a conversion per FMA"),
    "fused_db": KernelCost(
        "fused_db", "v3.2 + double buffer",
        fp32=2 * D + 2 * D + 6, sfu=1, tensor=0,
        smem_wf=(D / 8 * 2 + 2) / 32,
        lsu=0, dram_bytes_per_score=0, computes_masked=False,
        launch=LaunchShape(64, 161, 2 * 2 * 32 * 64 * 2 + 64 * 33 * 4, 64),
        notes="hides load latency; same instruction mix as v3.1"),
    "fused_tc": KernelCost(
        "fused_tc", "v4 tensor cores",
        fp32=12,               # 2 scale multiplies, max, mask selects, exp argument, row sum, P convert, O rescale
        sfu=1,
        tensor=4 * D,          # QK^T and PV on tensor cores
        # per warp and K/V tile (16 rows x 64 keys = 1024 scores), in 128-byte wavefronts:
        # K frags 64 + V frags 64 + S store 32 + S read twice 64 + P write 16 + O rescale 64
        # + O accumulator load/store 64 + P frags 64 = 432
        smem_wf=432 / 1024,
        lsu=0, dram_bytes_per_score=0, computes_masked=False,
        launch=LaunchShape(128, 80, 62464, 64),
        notes="shared-memory bound: S, P and O round-trip through shared memory (opaque WMMA layouts)"),
}

# FlashAttention-2 forward, as described in its paper (P and O stay in registers), for the
# reproduction study. smem: K and V tiles read once per 16-row warp slice via ldmatrix.
FA2_COST = dict(
    fp32=4,                    # exp argument FFMA, max, row-sum add, P pack (+ amortized O rescale)
    sfu=1,
    smem_wf_per_d=2 * 2 / 16 / 128,   # K and V bytes (2 B x d each) shared by 16 rows, per 128-B wavefront
)


# --------------------------------------------------------------------------- roofline
def scores(bh: int, n: int, causal: bool, computes_masked: bool) -> float:
    if not causal or computes_masked:
        return bh * n * n
    return bh * n * (n + 1) / 2


def useful_flops(bh: int, n: int, d: int = D, causal: bool = False) -> float:
    """FlashAttention convention: 4 N^2 d per head, halved for causal."""
    f = 4 * bh * n * n * d
    return f / 2 if causal else f


def bound(gpu: GPU, kernel: str, bh: int, n: int, causal: bool = False) -> dict:
    """Ideal time lower bound and which resource sets it."""
    c = COSTS[kernel]
    s = scores(bh, n, causal, c.computes_masked)
    hz = gpu.clock_ghz * 1e9
    per_sm = {  # cycles of each resource, summed over the whole problem, per SM
        "fp32": s * c.fp32 / gpu.fp32_lanes,
        "sfu": s * c.sfu / gpu.sfu_lanes,
        "tensor": s * c.tensor / gpu.tensor_flops_per_clk,
        "smem": s * c.smem_wf / gpu.smem_wavefronts_per_clk,
        "lsu": s * c.lsu / gpu.lsu_lanes,
    }
    t = {k: v / gpu.sms / hz for k, v in per_sm.items()}
    io_bytes = 4 * bh * n * D * 2 + s * c.dram_bytes_per_score   # Q, K, V read + O write (fp16) + N^2 buffers
    t["dram"] = io_bytes / (gpu.hbm_gbps * 1e9)
    limiter = max(t, key=t.get)
    time = t[limiter]
    return {"kernel": kernel, "gpu": gpu.name, "BH": bh, "N": n, "causal": causal,
            "time_ms": time * 1e3, "limiter": limiter,
            "tflops": useful_flops(bh, n, D, causal) / time / 1e12,
            **{f"t_{k}_ms": v * 1e3 for k, v in t.items()}, "dram_bytes": io_bytes}


def fa2_bounds(gpu: GPU, d: int) -> dict:
    """FlashAttention-2 forward throughput bounds per the same resource model.

    serial:     tensor-core, FP32 and SFU work execute one after another (no overlap)
    overlapped: they overlap perfectly; only the slowest resource counts
    """
    per_score = {
        "tensor": 4 * d / gpu.tensor_flops_per_clk,
        "fp32": FA2_COST["fp32"] / gpu.fp32_lanes,
        "sfu": FA2_COST["sfu"] / gpu.sfu_lanes,
        "smem": FA2_COST["smem_wf_per_d"] * d,
    }
    rate = gpu.sms * gpu.clock_ghz * 1e9  # SM-cycles per second
    serial = per_score["tensor"] + per_score["fp32"] + per_score["sfu"]
    overlapped = max(per_score.values())
    to_tflops = lambda cyc: 4 * d * rate / cyc / 1e12  # noqa: E731
    return {"d": d, "serial_tflops": to_tflops(serial), "overlapped_tflops": to_tflops(overlapped),
            "cycles_per_score": per_score}
