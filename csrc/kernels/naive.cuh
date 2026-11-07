#pragma once
// v1: naive attention.
// Three kernels; the full N x N score matrix S is written to HBM, read back, and written again.
// Written for clarity: every thread re-reads its operands straight from global memory.
#include "device_common.cuh"

namespace attn {

// S[bh, i, j] = scale * dot(Q[bh, i, :], K[bh, j, :])   (one thread per score)
template <typename T>
__global__ void scores_naive(const T* __restrict__ Q, const T* __restrict__ K, float* __restrict__ S,
                             int N, int d, float scale, bool causal) {
  const int bh = blockIdx.z;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || j >= N) return;
  const T* q = Q + ((size_t)bh * N + i) * d;
  const T* k = K + ((size_t)bh * N + j) * d;
  float acc = 0.f;
  for (int c = 0; c < d; ++c) acc += to_float(q[c]) * to_float(k[c]);
  float s = acc * scale;
  if (causal && j > i) s = -CUDART_INF_F;
  S[((size_t)bh * N + i) * N + j] = s;
}

// In-place row softmax: one thread walks an entire row three times (max, sum, normalize).
__global__ void softmax_naive(float* __restrict__ S, int N) {
  const int bh = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float* row = S + ((size_t)bh * N + i) * N;
  float m = -CUDART_INF_F;
  for (int j = 0; j < N; ++j) m = fmaxf(m, row[j]);
  float l = 0.f;
  for (int j = 0; j < N; ++j) l += expf(row[j] - m);
  const float inv = 1.f / l;
  for (int j = 0; j < N; ++j) row[j] = expf(row[j] - m) * inv;
}

// O[bh, i, c] = sum_j P[bh, i, j] * V[bh, j, c]   (one thread per output element)
template <typename T>
__global__ void pv_naive(const float* __restrict__ P, const T* __restrict__ V, T* __restrict__ O, int N, int d) {
  const int bh = blockIdx.z;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || c >= d) return;
  const float* p = P + ((size_t)bh * N + i) * N;
  const T* v = V + (size_t)bh * N * d;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += p[j] * to_float(v[(size_t)j * d + c]);
  O[((size_t)bh * N + i) * d + c] = from_float<T>(acc);
}

// S is caller-provided workspace of BH * N * N floats.
template <typename T>
cudaError_t launch_naive(const T* q, const T* k, const T* v, T* o, float* S, int BH, int N, int d, bool causal,
                         cudaStream_t stream) {
  const float scale = 1.f / sqrtf((float)d);
  const dim3 blk(16, 16);
  scores_naive<T><<<dim3(ceil_div(N, 16), ceil_div(N, 16), BH), blk, 0, stream>>>(q, k, S, N, d, scale, causal);
  softmax_naive<<<dim3(ceil_div(N, 256), BH), 256, 0, stream>>>(S, N);
  pv_naive<T><<<dim3(ceil_div(d, 16), ceil_div(N, 16), BH), blk, 0, stream>>>(S, v, o, N, d);
  return cudaGetLastError();
}

}  // namespace attn
