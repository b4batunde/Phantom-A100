// Standalone driver that runs one attention kernel with no PyTorch, so it can execute on the
// GPGPU-Sim cycle-level GPU simulator (or on a real GPU: the same binary works on both).
//
//   ./harness <kernel> <BH> <N> [causal=0] [seed=1]
//   kernel: naive | tiled | fused | fused_vec | fused_db | fused_tc | fused_tuned:<br>:<bc>:<db>
//
// Prints one machine-readable line:
//   RESULT kernel=... BH=... N=... causal=... max_abs_err=... status=PASS|FAIL
// The simulator writes cycle counts, DRAM traffic and occupancy to its own log; sim/run_sim.py
// parses both.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../csrc/kernels/naive.cuh"
#include "../csrc/kernels/tiled.cuh"
#include "../csrc/kernels/fused.cuh"
#include "../csrc/kernels/fused_tuned.cuh"
#include "../csrc/kernels/fused_tc.cuh"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::printf("RESULT status=CUDA_ERROR where=%s err=%s\n", #x, cudaGetErrorString(e_)); \
      return 2;                                                                            \
    }                                                                                      \
  } while (0)

// Double-precision reference: O = softmax(Q K^T / sqrt(d)) V for every (bh, row).
static void reference(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                      std::vector<float>& o, int BH, int N, int d, bool causal) {
  std::vector<double> s(N);
  const double scale = 1.0 / std::sqrt((double)d);
  for (int bh = 0; bh < BH; ++bh)
    for (int i = 0; i < N; ++i) {
      const float* qi = &q[((size_t)bh * N + i) * d];
      double m = -INFINITY;
      for (int j = 0; j < N; ++j) {
        if (causal && j > i) { s[j] = -INFINITY; continue; }
        const float* kj = &k[((size_t)bh * N + j) * d];
        double acc = 0;
        for (int c = 0; c < d; ++c) acc += (double)qi[c] * kj[c];
        s[j] = acc * scale;
        m = std::max(m, s[j]);
      }
      double l = 0;
      for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
      for (int c = 0; c < d; ++c) {
        double acc = 0;
        for (int j = 0; j < N; ++j) acc += s[j] * v[((size_t)bh * N + j) * d + c];
        o[((size_t)bh * N + i) * d + c] = (float)(acc / l);
      }
    }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <kernel> <BH> <N> [causal] [seed]\n", argv[0]);
    return 1;
  }
  const std::string name = argv[1];
  const int BH = std::atoi(argv[2]), N = std::atoi(argv[3]), d = 64;
  const bool causal = argc > 4 && std::atoi(argv[4]) != 0;
  const unsigned seed = argc > 5 ? (unsigned)std::atoi(argv[5]) : 1u;
  const size_t n = (size_t)BH * N * d;

  // FP16 inputs for every kernel (v1/v2/v3 also accept FP16), FP32 reference on the rounded values.
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<__half> hq(n), hk(n), hv(n), ho(n);
  std::vector<float> fq(n), fk(n), fv(n), ref(n);
  for (size_t i = 0; i < n; ++i) {
    hq[i] = __float2half(dist(rng)); fq[i] = __half2float(hq[i]);
    hk[i] = __float2half(dist(rng)); fk[i] = __half2float(hk[i]);
    hv[i] = __float2half(dist(rng)); fv[i] = __half2float(hv[i]);
  }

  __half *q, *k, *v, *o;
  float* S = nullptr;
  CK(cudaMalloc(&q, n * sizeof(__half)));
  CK(cudaMalloc(&k, n * sizeof(__half)));
  CK(cudaMalloc(&v, n * sizeof(__half)));
  CK(cudaMalloc(&o, n * sizeof(__half)));
  CK(cudaMemcpy(q, hq.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(k, hk.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v, hv.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CK(cudaMemset(o, 0, n * sizeof(__half)));

  cudaError_t err;
  if (name == "naive" || name == "tiled") {
    CK(cudaMalloc(&S, (size_t)BH * N * N * sizeof(float)));
    err = name == "naive" ? attn::launch_naive<__half>(q, k, v, o, S, BH, N, d, causal, 0)
                          : attn::launch_tiled<__half>(q, k, v, o, S, BH, N, d, causal, 0);
  } else if (name == "fused") {
    err = attn::launch_fused<__half>(q, k, v, o, BH, N, causal, 0);
  } else if (name == "fused_vec" || name == "fused_db") {
    err = attn::launch_fused_tuned(q, k, v, o, BH, N, causal, name == "fused_db", 64, 32, 0);
  } else if (name.rfind("fused_tuned:", 0) == 0) {
    int br = 64, bc = 32, db = 0;
    std::sscanf(name.c_str(), "fused_tuned:%d:%d:%d", &br, &bc, &db);
    err = attn::launch_fused_tuned(q, k, v, o, BH, N, causal, db != 0, br, bc, 0);
  } else if (name == "fused_tc") {
    err = attn::launch_fused_tc(q, k, v, o, BH, N, causal, 0);
  } else {
    std::fprintf(stderr, "unknown kernel %s\n", name.c_str());
    return 1;
  }
  CK(err);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho.data(), o, n * sizeof(__half), cudaMemcpyDeviceToHost));

  reference(fq, fk, fv, ref, BH, N, d, causal);
  double max_err = 0;
  for (size_t i = 0; i < n; ++i) {
    const double e = std::fabs((double)__half2float(ho[i]) - ref[i]);
    max_err = std::isfinite(e) ? std::max(max_err, e) : INFINITY;
  }
  const bool pass = max_err <= 2e-3;
  std::printf("RESULT kernel=%s BH=%d N=%d causal=%d max_abs_err=%.3e status=%s\n", name.c_str(), BH, N,
              (int)causal, max_err, pass ? "PASS" : "FAIL");
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
  if (S) cudaFree(S);
  return pass ? 0 : 3;
}
