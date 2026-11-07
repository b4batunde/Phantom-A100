#pragma once
// v4: fused attention on tensor cores (WMMA, FP16 in / FP32 accumulate, head dim 64).
//
// v3 does every multiply-add on the regular FP32 units, one thread per query row. That caps it
// far below FlashAttention-2, which runs both matmuls on tensor cores. v4 keeps the same online
// softmax but moves the two matmuls onto tensor cores through the WMMA API:
//
//   block = 4 warps = 64 query rows (16 per warp); K/V tiles of 64 keys are shared by the block.
//   per warp, per K/V tile:
//     1. S (16 x 64)  = Q_w (16 x 64) @ K_j^T      4 x 4 m16n16k16 MMAs, Q fragments kept in registers
//     2. online softmax on S in shared memory      2 lanes per row, shuffles for row max / row sum
//        P (fp16) written to shared memory, O rows rescaled by alpha = 2^(m_old - m_new)
//     3. O (16 x 64) += P (16 x 64) @ V_j          4 x 4 MMAs, O accumulator round-trips through smem
//   finally O / l is written to HBM.
//
// WMMA fragments have an unspecified register layout, so every per-row operation (masking,
// softmax, rescaling) happens on shared-memory copies. That costs bandwidth that a hand-written
// mma.sync kernel avoids; it is the next step after this one.
#include <mma.h>
#include "device_common.cuh"

namespace attn {
namespace tc {


constexpr int D = 64;       // head dimension
constexpr int BR = 64;      // query rows per block
constexpr int BC = 64;      // keys per K/V tile
constexpr int WARPS = 4;    // 16 query rows per warp
constexpr int THREADS = WARPS * 32;
constexpr int WM = 16;      // WMMA tile edge

// Padded leading dimensions (halves: multiple of 8, floats: multiple of 4) to spread banks while
// keeping every WMMA pointer 32-byte aligned.
constexpr int LDK = D + 8;   // K, V tiles (half)
constexpr int LDP = BC + 8;  // P / staged Q (half)
constexpr int LDS = BC + 4;  // S (float)
constexpr int LDO = D + 4;   // O accumulator (float)

constexpr size_t KV_BYTES = 2 * BC * LDK * sizeof(__half);
constexpr size_t WARP_BYTES = WM * LDS * sizeof(float) + WM * LDP * sizeof(__half) + WM * LDO * sizeof(float);
constexpr size_t SMEM_BYTES = KV_BYTES + WARPS * WARP_BYTES;
static_assert(LDP >= D, "staged Q reuses the P buffer");

__global__ void __launch_bounds__(THREADS)
fused_tc_fwd(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
             __half* __restrict__ O, int N, float scale_log2, bool causal) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* Ks = reinterpret_cast<__half*>(smem_raw);  // [BC][LDK]
  __half* Vs = Ks + BC * LDK;                        // [BC][LDK]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  unsigned char* wbase = smem_raw + KV_BYTES + warp * WARP_BYTES;
  float* Sw = reinterpret_cast<float*>(wbase);                                   // [16][LDS]
  __half* Pw = reinterpret_cast<__half*>(wbase + WM * LDS * sizeof(float));      // [16][LDP]
  float* Ow = reinterpret_cast<float*>(wbase + WM * LDS * sizeof(float) + WM * LDP * sizeof(__half));  // [16][LDO]

  const int bh = blockIdx.y;
  const int i0 = blockIdx.x * BR;
  const int w0 = i0 + warp * WM;  // first query row of this warp
  const size_t base = (size_t)bh * N * D;

  // Each lane owns half of one row for all per-row work: row r, columns [h*32, h*32 + 32).
  const int r = lane >> 1, h = lane & 1;
  const int row = w0 + r;

  // ---- stage this warp's 16 Q rows in smem (zero padded) and load them as A fragments
  for (int chunk = lane; chunk < WM * (D / 8); chunk += 32) {
    const int rr = chunk / (D / 8), c = (chunk % (D / 8)) * 8;
    const int gi = w0 + rr;
    *reinterpret_cast<uint4*>(Pw + rr * LDP + c) =
        gi < N ? *reinterpret_cast<const uint4*>(Q + base + (size_t)gi * D + c) : make_uint4(0, 0, 0, 0);
  }
  __syncwarp();
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WM, WM, __half, nvcuda::wmma::row_major> q_frag[D / WM];
#pragma unroll
  for (int kk = 0; kk < D / WM; ++kk) nvcuda::wmma::load_matrix_sync(q_frag[kk], Pw + kk * WM, LDP);

  // ---- zero this lane's half-row of O
#pragma unroll
  for (int c = 0; c < 32; ++c) Ow[r * LDO + h * 32 + c] = 0.f;
  float m = -CUDART_INF_F, l = 0.f;  // row state, duplicated in the two lanes of a row

  const int kv_end = causal ? min(N, i0 + BR) : N;
  for (int j0 = 0; j0 < kv_end; j0 += BC) {
    // ---- 0. block-wide load of K_j, V_j (128-bit, zero padded past N)
    __syncthreads();  // previous tile fully consumed
    for (int chunk = threadIdx.x; chunk < BC * (D / 8); chunk += THREADS) {
      const int rr = chunk / (D / 8), c = (chunk % (D / 8)) * 8;
      const int jj = j0 + rr;
      const uint4 zero = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(Ks + rr * LDK + c) =
          jj < N ? *reinterpret_cast<const uint4*>(K + base + (size_t)jj * D + c) : zero;
      *reinterpret_cast<uint4*>(Vs + rr * LDK + c) =
          jj < N ? *reinterpret_cast<const uint4*>(V + base + (size_t)jj * D + c) : zero;
    }
    __syncthreads();

    // Causal: a tile that starts after this warp's last row contributes nothing.
    if (causal && j0 > w0 + WM - 1) continue;

    // ---- 1. S = Q_w K_j^T on tensor cores. K is [key][d] row-major == K^T col-major.
#pragma unroll
    for (int nt = 0; nt < BC / WM; ++nt) {
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WM, WM, float> s_frag;
      nvcuda::wmma::fill_fragment(s_frag, 0.f);
#pragma unroll
      for (int kk = 0; kk < D / WM; ++kk) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WM, WM, __half, nvcuda::wmma::col_major> k_frag;
        nvcuda::wmma::load_matrix_sync(k_frag, Ks + (nt * WM) * LDK + kk * WM, LDK);
        nvcuda::wmma::mma_sync(s_frag, q_frag[kk], k_frag, s_frag);
      }
      nvcuda::wmma::store_matrix_sync(Sw + nt * WM, s_frag, LDS, nvcuda::wmma::mem_row_major);
    }
    __syncwarp();

    // ---- 2. online softmax, two lanes per row
    const float* srow = Sw + r * LDS + h * 32;
    float mx = -CUDART_INF_F;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int col = j0 + h * 32 + c;
      const bool masked = col >= N || (causal && col > row);
      mx = fmaxf(mx, masked ? -CUDART_INF_F : srow[c] * scale_log2);
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
    const float m_new = fmaxf(m, mx);  // finite: column 0 of tile 0 is always visible
    const float alpha = exp2f(m - m_new);

    __half* prow = Pw + r * LDP + h * 32;
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int col = j0 + h * 32 + c;
      const bool masked = col >= N || (causal && col > row);
      const float p = masked ? 0.f : exp2f(srow[c] * scale_log2 - m_new);
      prow[c] = __float2half(p);
      sum += p;
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    l = l * alpha + sum;
    m = m_new;
    float* orow = Ow + r * LDO + h * 32;
#pragma unroll
    for (int c = 0; c < 32; ++c) orow[c] *= alpha;
    __syncwarp();

    // ---- 3. O += P V_j on tensor cores (accumulator loaded from and stored back to smem)
#pragma unroll
    for (int nt = 0; nt < D / WM; ++nt) {
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WM, WM, float> o_frag;
      nvcuda::wmma::load_matrix_sync(o_frag, Ow + nt * WM, LDO, nvcuda::wmma::mem_row_major);
#pragma unroll
      for (int kk = 0; kk < BC / WM; ++kk) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WM, WM, __half, nvcuda::wmma::row_major> p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WM, WM, __half, nvcuda::wmma::row_major> v_frag;
        nvcuda::wmma::load_matrix_sync(p_frag, Pw + kk * WM, LDP);
        nvcuda::wmma::load_matrix_sync(v_frag, Vs + (kk * WM) * LDK + nt * WM, LDK);
        nvcuda::wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
      }
      nvcuda::wmma::store_matrix_sync(Ow + nt * WM, o_frag, LDO, nvcuda::wmma::mem_row_major);
    }
    __syncwarp();
  }

  // ---- write O / l, 128-bit stores (4 per lane: 32 halves)
  if (row < N) {
    const float inv = 1.f / l;
    const float* orow = Ow + r * LDO + h * 32;
    __half* out = O + base + (size_t)row * D + h * 32;
#pragma unroll
    for (int c = 0; c < 32; c += 8) {
      uint4 raw;
      __half2* h2 = reinterpret_cast<__half2*>(&raw);
#pragma unroll
      for (int t = 0; t < 4; ++t) h2[t] = __floats2half2_rn(orow[c + 2 * t] * inv, orow[c + 2 * t + 1] * inv);
      *reinterpret_cast<uint4*>(out + c) = raw;
    }
  }
}


}  // namespace tc

constexpr size_t fused_tc_smem_bytes() { return tc::SMEM_BYTES; }

// Requires compute capability 7.0+ and fused_tc_smem_bytes() of opt-in shared memory; d must be 64.
inline cudaError_t launch_fused_tc(const __half* q, const __half* k, const __half* v, __half* o, int BH, int N,
                                   bool causal, cudaStream_t stream) {
  cudaError_t e = cudaFuncSetAttribute(tc::fused_tc_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)tc::SMEM_BYTES);
  if (e != cudaSuccess) return e;
  const float scale_log2 = kLog2e / sqrtf((float)tc::D);
  tc::fused_tc_fwd<<<dim3(ceil_div(N, tc::BR), BH), tc::THREADS, tc::SMEM_BYTES, stream>>>(q, k, v, o, N, scale_log2,
                                                                                          causal);
  return cudaGetLastError();
}

}  // namespace attn
