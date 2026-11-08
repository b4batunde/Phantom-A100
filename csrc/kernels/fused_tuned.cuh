#pragma once
// v3.1 / v3.2: tuned fused attention (FP16 only, head dim 64).
//
// Same algorithm and thread mapping as v3 (one thread per query row, online softmax in the
// log2 domain). Changes, each one benchmarked on its own:
//
//   v3.1  VECTORIZED LOADS   K/V tiles are copied with 128-bit (8 x half) loads and kept in
//                            shared memory as FP16, halving shared-memory footprint and
//                            global-load instruction count. Q and O use 128-bit accesses too.
//   v3.2  DOUBLE BUFFERING   two K/V stages; tile t+1 is copied with cp.async (sm_80+) while
//                            tile t is being computed, hiding global-memory latency.
//   sweep TILE SIZES         BR (query rows per block) x BC (keys per tile) are template
//                            parameters; bench/sweep_tiles.py picks the best per GPU.
//
// Shared memory is dynamic so larger tiles can opt in above the 48 KB static limit.
#include "device_common.cuh"

namespace attn {


constexpr int TUNED_D = 64;      // head dimension
constexpr int CHUNK = 8;         // halves per 128-bit transfer
constexpr int CHUNKS_PER_ROW = TUNED_D / CHUNK;

// ------------------------------------------------------------------ async copy helpers
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_bytes = valid ? 16 : 0;  // 0 => zero-fill the 16 destination bytes
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src), "r"(src_bytes));
#else
  *reinterpret_cast<uint4*>(smem_dst) = valid ? *reinterpret_cast<const uint4*>(gmem_src) : make_uint4(0, 0, 0, 0);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N_PENDING>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N_PENDING));
#endif
}

// Copy rows [j0, j0 + BC) of K and V into one shared-memory stage (rows past N are zeros).
template <int BR, int BC, bool ASYNC>
__device__ __forceinline__ void load_kv_tile(__half* Ks, __half* Vs, const __half* Kb, const __half* Vb,
                                             int j0, int N, int tid) {
#pragma unroll
  for (int chunk = tid; chunk < BC * CHUNKS_PER_ROW; chunk += BR) {
    const int r = chunk / CHUNKS_PER_ROW, c = (chunk % CHUNKS_PER_ROW) * CHUNK;
    const int jj = j0 + r;
    const bool valid = jj < N;
    const size_t src = (size_t)(valid ? jj : 0) * TUNED_D + c;  // keep the address legal when zero-filling
    if (ASYNC) {
      cp_async_16(Ks + r * TUNED_D + c, Kb + src, valid);
      cp_async_16(Vs + r * TUNED_D + c, Vb + src, valid);
    } else {
      const uint4 zero = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(Ks + r * TUNED_D + c) = valid ? *reinterpret_cast<const uint4*>(Kb + src) : zero;
      *reinterpret_cast<uint4*>(Vs + r * TUNED_D + c) = valid ? *reinterpret_cast<const uint4*>(Vb + src) : zero;
    }
  }
}

// ------------------------------------------------------------------ kernel
template <int BR, int BC, bool DOUBLE_BUFFER, int MINB = 0>  // MINB 0: no min-blocks hint
__global__ void __launch_bounds__(BR, MINB)
fused_tuned_fwd(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
                __half* __restrict__ O, int N, float scale_log2, bool causal) {
  constexpr int STAGES = DOUBLE_BUFFER ? 2 : 1;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* Ks = reinterpret_cast<__half*>(smem_raw);           // [STAGES][BC][TUNED_D]
  __half* Vs = Ks + STAGES * BC * TUNED_D;                           // [STAGES][BC][TUNED_D]
  float* Ss = reinterpret_cast<float*>(Vs + STAGES * BC * TUNED_D);  // [BR][BC + 1]

  const int tid = threadIdx.x;
  const int bh = blockIdx.y;
  const int i0 = blockIdx.x * BR;
  const int i = i0 + tid;
  const bool valid = i < N;
  const size_t base = (size_t)bh * N * TUNED_D;
  const __half* Kb = K + base;
  const __half* Vb = V + base;

  // Query row: 8 x 128-bit loads, pre-scaled into the log2 domain.
  float q[TUNED_D], o[TUNED_D];
#pragma unroll
  for (int c = 0; c < TUNED_D; c += CHUNK) {
    uint4 raw = valid ? *reinterpret_cast<const uint4*>(Q + base + (size_t)i * TUNED_D + c) : make_uint4(0, 0, 0, 0);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int t = 0; t < CHUNK / 2; ++t) {
      const float2 f = __half22float2(h2[t]);
      q[c + 2 * t] = f.x * scale_log2;
      q[c + 2 * t + 1] = f.y * scale_log2;
    }
  }
#pragma unroll
  for (int c = 0; c < TUNED_D; ++c) o[c] = 0.f;
  float m = -CUDART_INF_F, l = 0.f;
  float* Srow = Ss + tid * (BC + 1);

  const int kv_end = causal ? min(N, i0 + BR) : N;
  const int num_tiles = (kv_end + BC - 1) / BC;

  if (DOUBLE_BUFFER) {  // prologue: start tile 0
    load_kv_tile<BR, BC, true>(Ks, Vs, Kb, Vb, 0, N, tid);
    cp_async_commit();
  }

  for (int t = 0; t < num_tiles; ++t) {
    const int j0 = t * BC;
    const int stage = DOUBLE_BUFFER ? (t & 1) : 0;
    const __half* Kt = Ks + stage * BC * TUNED_D;
    const __half* Vt = Vs + stage * BC * TUNED_D;

    if (DOUBLE_BUFFER) {
      if (t + 1 < num_tiles) {  // prefetch the next tile into the other stage
        const int nxt = (t + 1) & 1;
        load_kv_tile<BR, BC, true>(Ks + nxt * BC * TUNED_D, Vs + nxt * BC * TUNED_D, Kb, Vb, j0 + BC, N, tid);
        cp_async_commit();
        cp_async_wait<1>();  // everything except the prefetch just issued has landed
      } else {
        cp_async_wait<0>();
      }
    } else {
      load_kv_tile<BR, BC, false>(Ks, Vs, Kb, Vb, j0, N, tid);
    }
    __syncthreads();

    // Scores for this row; K is read as half2 (broadcast: every thread reads the same address).
    float m_tile = -CUDART_INF_F;
    for (int jj = 0; jj < BC; ++jj) {
      const __half2* k2 = reinterpret_cast<const __half2*>(Kt + jj * TUNED_D);
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < TUNED_D / 2; ++c) {
        const float2 kf = __half22float2(k2[c]);
        s += q[2 * c] * kf.x + q[2 * c + 1] * kf.y;
      }
      const int col = j0 + jj;
      if (col >= N || (causal && col > i)) s = -CUDART_INF_F;
      Srow[jj] = s;
      m_tile = fmaxf(m_tile, s);
    }

    const float m_new = fmaxf(m, m_tile);
    if (m_new != -CUDART_INF_F) {
      const float alpha = exp2f(m - m_new);
      l *= alpha;
#pragma unroll
      for (int c = 0; c < TUNED_D; ++c) o[c] *= alpha;
      for (int jj = 0; jj < BC; ++jj) {
        const float p = exp2f(Srow[jj] - m_new);
        l += p;
        const __half2* v2 = reinterpret_cast<const __half2*>(Vt + jj * TUNED_D);
#pragma unroll
        for (int c = 0; c < TUNED_D / 2; ++c) {
          const float2 vf = __half22float2(v2[c]);
          o[2 * c] += p * vf.x;
          o[2 * c + 1] += p * vf.y;
        }
      }
      m = m_new;
    }
    __syncthreads();  // this stage may be overwritten by the next prefetch
  }

  if (valid) {  // 8 x 128-bit stores
    const float inv = 1.f / l;
    __half* out = O + base + (size_t)i * TUNED_D;
#pragma unroll
    for (int c = 0; c < TUNED_D; c += CHUNK) {
      uint4 raw;
      __half2* h2 = reinterpret_cast<__half2*>(&raw);
#pragma unroll
      for (int t = 0; t < CHUNK / 2; ++t) h2[t] = __floats2half2_rn(o[c + 2 * t] * inv, o[c + 2 * t + 1] * inv);
      *reinterpret_cast<uint4*>(out + c) = raw;
    }
  }
}


// Shared memory needed by one block (dynamic).
constexpr size_t fused_tuned_smem_bytes(int br, int bc, bool double_buffer) {
  return 2 * (double_buffer ? 2 : 1) * (size_t)bc * TUNED_D * sizeof(__half) + (size_t)br * (bc + 1) * sizeof(float);
}

template <int BR, int BC, bool DB, int MINB = 0>
cudaError_t launch_fused_tuned_cfg(const __half* q, const __half* k, const __half* v, __half* o, int BH, int N,
                                   bool causal, cudaStream_t stream) {
  constexpr size_t smem = fused_tuned_smem_bytes(BR, BC, DB);
  auto kernel = fused_tuned_fwd<BR, BC, DB, MINB>;
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
  }
  const float scale_log2 = kLog2e / sqrtf((float)TUNED_D);
  kernel<<<dim3(ceil_div(N, BR), BH), BR, smem, stream>>>(q, k, v, o, N, scale_log2, causal);
  return cudaGetLastError();
}

// Configurations compiled for the tile sweep. Add a line here to try another one.
#define ATTN_TUNED_CONFIGS(X) X(64, 32) X(64, 64) X(128, 32) X(128, 64)

// Returns cudaErrorInvalidValue if (br, bc, maxreg) is not a compiled configuration.
// maxreg = 0: no register cap (the original v3.1 / v3.2). maxreg = 128: __launch_bounds__ asks ptxas
// for at most 128 registers per thread (used by autotune/; may spill on hardware).
inline cudaError_t launch_fused_tuned(const __half* q, const __half* k, const __half* v, __half* o, int BH, int N,
                                      bool causal, bool double_buffer, int br, int bc, cudaStream_t stream,
                                      int maxreg = 0) {
#define ATTN_TRY_MB(BR_, BC_, MB_)                                                                         \
  return double_buffer ? launch_fused_tuned_cfg<BR_, BC_, true, MB_>(q, k, v, o, BH, N, causal, stream)    \
                       : launch_fused_tuned_cfg<BR_, BC_, false, MB_>(q, k, v, o, BH, N, causal, stream);
#define ATTN_TRY(BR_, BC_)                                                                                 \
  if (br == BR_ && bc == BC_) {                                                                            \
    if (maxreg == 0) { ATTN_TRY_MB(BR_, BC_, 0) }                                                          \
    if (maxreg == 128) { ATTN_TRY_MB(BR_, BC_, 65536 / (BR_ * 128)) }                                      \
    return cudaErrorInvalidValue;                                                                          \
  }
  ATTN_TUNED_CONFIGS(ATTN_TRY)
#undef ATTN_TRY
#undef ATTN_TRY_MB
  return cudaErrorInvalidValue;
}

}  // namespace attn
