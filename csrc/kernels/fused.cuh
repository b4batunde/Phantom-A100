#pragma once
// v3: fused attention with online softmax (FlashAttention-style forward pass).
//
// One thread block owns BR query rows (one thread per row). It streams K and V through
// shared memory in tiles of BC rows and folds each score tile into a running output:
//
//   m_new = max(m, rowmax(S_ij))
//   l     = exp(m - m_new) * l + rowsum(exp(S_ij - m_new))
//   O     = exp(m - m_new) * O + exp(S_ij - m_new) @ V_j
//
// After the last tile O / l is written. The N x N score matrix never touches HBM,
// so global traffic is O(N * d) instead of O(N^2).
//
// Scores are kept in the log2 domain (Q is pre-scaled by log2(e) / sqrt(d)) so the
// kernel can use exp2f, which maps to a single hardware instruction.
//
// Deliberately left for the tuning phase (each one benchmarked separately):
//   * vectorized 128-bit / half2 loads
//   * double-buffered K/V tiles (cp.async on sm_80+)
//   * more than one thread per row, larger BR, tensor cores (mma.sync)
#include "device_common.cuh"

namespace attn {


template <typename T, int D, int BR, int BC, int MINB = 0>  // MINB 0: no min-blocks hint (same PTX as plain __launch_bounds__(BR))
__global__ void __launch_bounds__(BR, MINB)
fused_fwd(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
          int N, float scale_log2, bool causal) {
  __shared__ float Ks[BC][D];       // every thread reads the same Ks[j][c] -> broadcast, no padding needed
  __shared__ float Vs[BC][D];
  __shared__ float Ss[BR][BC + 1];  // per-thread score row; +1 padding keeps the 32 threads of a warp on 32 banks

  const int tid = threadIdx.x;
  const int bh = blockIdx.y;
  const int i0 = blockIdx.x * BR;
  const int i = i0 + tid;
  const bool valid = i < N;

  const T* Qb = Q + (size_t)bh * N * D;
  const T* Kb = K + (size_t)bh * N * D;
  const T* Vb = V + (size_t)bh * N * D;

  // Query row and output accumulator live in registers.
  float q[D], o[D];
#pragma unroll
  for (int c = 0; c < D; ++c) {
    q[c] = valid ? to_float(Qb[(size_t)i * D + c]) * scale_log2 : 0.f;
    o[c] = 0.f;
  }
  float m = -CUDART_INF_F, l = 0.f;

  // Causal: no key beyond the block's last query row can contribute.
  const int kv_end = causal ? min(N, i0 + BR) : N;

  for (int j0 = 0; j0 < kv_end; j0 += BC) {
    // 1. Cooperative, coalesced load of K_j and V_j into shared memory.
    for (int idx = tid; idx < BC * D; idx += BR) {
      const int r = idx / D, c = idx % D, jj = j0 + r;
      Ks[r][c] = jj < N ? to_float(Kb[(size_t)jj * D + c]) : 0.f;
      Vs[r][c] = jj < N ? to_float(Vb[(size_t)jj * D + c]) : 0.f;
    }
    __syncthreads();

    // 2. Score tile for this row, plus its max.
    float m_tile = -CUDART_INF_F;
    for (int jj = 0; jj < BC; ++jj) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < D; ++c) s += q[c] * Ks[jj][c];
      const int col = j0 + jj;
      if (col >= N || (causal && col > i)) s = -CUDART_INF_F;
      Ss[tid][jj] = s;
      m_tile = fmaxf(m_tile, s);
    }

    // 3-4. Online softmax update: rescale the old state, then add this tile.
    const float m_new = fmaxf(m, m_tile);
    if (m_new != -CUDART_INF_F) {
      const float alpha = exp2f(m - m_new);  // m == -inf on the first tile -> alpha = 0
      l *= alpha;
#pragma unroll
      for (int c = 0; c < D; ++c) o[c] *= alpha;
      for (int jj = 0; jj < BC; ++jj) {
        const float p = exp2f(Ss[tid][jj] - m_new);
        l += p;
#pragma unroll
        for (int c = 0; c < D; ++c) o[c] += p * Vs[jj][c];
      }
      m = m_new;
    }
    __syncthreads();  // before the next tile overwrites Ks / Vs
  }

  // 5. Normalize and write the output row.
  if (valid) {
    const float inv = 1.f / l;
    T* out = O + ((size_t)bh * N + i) * D;
#pragma unroll
    for (int c = 0; c < D; ++c) out[c] = from_float<T>(o[c] * inv);
  }
}


// Only head dimension 64 is compiled; the caller checks d == 64.
template <typename T>
cudaError_t launch_fused(const T* q, const T* k, const T* v, T* o, int BH, int N, bool causal, cudaStream_t stream) {
  constexpr int D = 64, BR = 64, BC = 32;
  const float scale_log2 = kLog2e / sqrtf((float)D);
  fused_fwd<T, D, BR, BC><<<dim3(ceil_div(N, BR), BH), BR, 0, stream>>>(q, k, v, o, N, scale_log2, causal);
  return cudaGetLastError();
}

// Design-space entry point used by autotune/ (sim/harness.cu): v3 at another query-tile size
// and/or with a register cap. maxreg = 0 means no cap; otherwise __launch_bounds__ asks ptxas to fit
// 65536 / (BR * maxreg) blocks per SM, i.e. at most `maxreg` registers per thread. In the simulator
// a cap is free (PTX has no spills); on hardware ptxas may spill to local memory.
template <int BR, int MINB>
cudaError_t launch_fused_cfg_t(const __half* q, const __half* k, const __half* v, __half* o, int BH, int N,
                               bool causal, cudaStream_t stream) {
  constexpr int D = 64, BC = 32;
  const float scale_log2 = kLog2e / sqrtf((float)D);
  fused_fwd<__half, D, BR, BC, MINB><<<dim3(ceil_div(N, BR), BH), BR, 0, stream>>>(q, k, v, o, N, scale_log2, causal);
  return cudaGetLastError();
}

inline cudaError_t launch_fused_cfg(const __half* q, const __half* k, const __half* v, __half* o, int BH, int N,
                                    bool causal, int br, int maxreg, cudaStream_t stream) {
  if (br == 64 && maxreg == 0) return launch_fused_cfg_t<64, 0>(q, k, v, o, BH, N, causal, stream);
  if (br == 64 && maxreg == 128) return launch_fused_cfg_t<64, 8>(q, k, v, o, BH, N, causal, stream);
  if (br == 128 && maxreg == 0) return launch_fused_cfg_t<128, 0>(q, k, v, o, BH, N, causal, stream);
  if (br == 128 && maxreg == 128) return launch_fused_cfg_t<128, 4>(q, k, v, o, BH, N, causal, stream);
  return cudaErrorInvalidValue;
}

}  // namespace attn
