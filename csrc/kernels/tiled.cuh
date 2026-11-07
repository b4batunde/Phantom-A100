#pragma once
// v2: tiled attention.
// Same three steps as v1, but:
//   * both matmuls stage 32 x 32 tiles of their operands in shared memory and reuse them,
//     so each global element is read once per tile instead of once per output;
//   * shared tiles are padded to 33 columns so column-wise reads hit 32 different banks;
//   * softmax uses one warp per row, a single online (max, sum) pass and shuffle reductions.
// S is still materialized in HBM; v3 removes it.
#include "device_common.cuh"

namespace attn {

constexpr int TILE = 32;

// S tile (i0..i0+31, j0..j0+31) = scale * Q_tile @ K_tile^T
template <typename T>
__global__ void __launch_bounds__(TILE * TILE)
scores_tiled(const T* __restrict__ Q, const T* __restrict__ K, float* __restrict__ S,
             int N, int d, float scale, bool causal) {
  __shared__ float Qs[TILE][TILE + 1];
  __shared__ float Ks[TILE][TILE + 1];
  const int bh = blockIdx.z;
  const int i0 = blockIdx.y * TILE, j0 = blockIdx.x * TILE;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const T* Qb = Q + (size_t)bh * N * d;
  const T* Kb = K + (size_t)bh * N * d;

  float acc = 0.f;
  for (int k0 = 0; k0 < d; k0 += TILE) {
    // Coalesced loads: consecutive tx read consecutive elements of one row.
    const int c = k0 + tx;
    Qs[ty][tx] = (i0 + ty < N && c < d) ? to_float(Qb[(size_t)(i0 + ty) * d + c]) : 0.f;
    Ks[ty][tx] = (j0 + ty < N && c < d) ? to_float(Kb[(size_t)(j0 + ty) * d + c]) : 0.f;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) acc += Qs[ty][kk] * Ks[tx][kk];  // Qs: broadcast, Ks: conflict-free (padding)
    __syncthreads();
  }
  const int i = i0 + ty, j = j0 + tx;
  if (i < N && j < N) {
    float s = acc * scale;
    if (causal && j > i) s = -CUDART_INF_F;
    S[((size_t)bh * N + i) * N + j] = s;
  }
}

// Merge two partial softmax states (m, l) where l is a sum of exp(x - m).
__device__ __forceinline__ void merge_ml(float& m, float& l, float m2, float l2) {
  const float mn = fmaxf(m, m2);
  if (mn == -CUDART_INF_F) return;  // both empty
  l = l * __expf(m - mn) + l2 * __expf(m2 - mn);
  m = mn;
}

// One warp per row. Pass 1: online max/sum per lane, then a warp-shuffle merge. Pass 2: normalize.
__global__ void softmax_warp(float* __restrict__ S, int N) {
  constexpr int WARPS = 8;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int bh = blockIdx.y;
  const int i = blockIdx.x * WARPS + warp;
  if (i >= N) return;
  float* row = S + ((size_t)bh * N + i) * N;

  float m = -CUDART_INF_F, l = 0.f;
  for (int j = lane; j < N; j += 32) {
    const float x = row[j];
    if (x > m) { l = l * __expf(m - x) + 1.f; m = x; }
    else if (x != -CUDART_INF_F) { l += __expf(x - m); }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float m2 = __shfl_xor_sync(0xffffffff, m, off);
    const float l2 = __shfl_xor_sync(0xffffffff, l, off);
    merge_ml(m, l, m2, l2);
  }
  const float inv = 1.f / l;
  for (int j = lane; j < N; j += 32) row[j] = __expf(row[j] - m) * inv;
}

// O tile (i0..i0+31, c0..c0+31) = P_tile_rows @ V_tile_cols
template <typename T>
__global__ void __launch_bounds__(TILE * TILE)
pv_tiled(const float* __restrict__ P, const T* __restrict__ V, T* __restrict__ O, int N, int d, bool causal) {
  __shared__ float Ps[TILE][TILE + 1];
  __shared__ float Vs[TILE][TILE + 1];
  const int bh = blockIdx.z;
  const int i0 = blockIdx.y * TILE, c0 = blockIdx.x * TILE;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const float* Pb = P + (size_t)bh * N * N;
  const T* Vb = V + (size_t)bh * N * d;
  // With a causal mask, P is zero to the right of the diagonal: skip those tiles.
  const int k_end = causal ? min(N, i0 + TILE) : N;

  float acc = 0.f;
  for (int k0 = 0; k0 < k_end; k0 += TILE) {
    Ps[ty][tx] = (i0 + ty < N && k0 + tx < N) ? Pb[(size_t)(i0 + ty) * N + k0 + tx] : 0.f;
    Vs[ty][tx] = (k0 + ty < N && c0 + tx < d) ? to_float(Vb[(size_t)(k0 + ty) * d + c0 + tx]) : 0.f;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) acc += Ps[ty][kk] * Vs[kk][tx];
    __syncthreads();
  }
  const int i = i0 + ty, c = c0 + tx;
  if (i < N && c < d) O[((size_t)bh * N + i) * d + c] = from_float<T>(acc);
}

// S is caller-provided workspace of BH * N * N floats.
template <typename T>
cudaError_t launch_tiled(const T* q, const T* k, const T* v, T* o, float* S, int BH, int N, int d, bool causal,
                         cudaStream_t stream) {
  const float scale = 1.f / sqrtf((float)d);
  const dim3 blk(TILE, TILE);
  scores_tiled<T><<<dim3(ceil_div(N, TILE), ceil_div(N, TILE), BH), blk, 0, stream>>>(q, k, S, N, d, scale, causal);
  softmax_warp<<<dim3(ceil_div(N, 8), BH), 256, 0, stream>>>(S, N);
  pv_tiled<T><<<dim3(ceil_div(d, TILE), ceil_div(N, TILE), BH), blk, 0, stream>>>(S, v, o, N, d, causal);
  return cudaGetLastError();
}

}  // namespace attn
