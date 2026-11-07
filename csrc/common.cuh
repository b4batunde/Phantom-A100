#pragma once
// PyTorch-side helpers. Kernels themselves live in csrc/kernels/ and do not depend on PyTorch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "kernels/device_common.cuh"

using attn::ceil_div;

// Raw device pointer of a tensor, viewed as the CUDA storage type T (float or __half).
template <typename T> inline T* dptr(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

// Runs the lambda body with `T` bound to float or __half depending on the tensor dtype.
#define DISPATCH_FLOAT_HALF(STYPE, NAME, ...)                                  \
  [&] {                                                                        \
    if ((STYPE) == at::kFloat) { using T = float; __VA_ARGS__(); }             \
    else if ((STYPE) == at::kHalf) { using T = __half; __VA_ARGS__(); }        \
    else { TORCH_CHECK(false, NAME, ": only float32 and float16 are supported"); } \
  }()

#define ATTN_CHECK(EXPR, NAME)                                                         \
  do {                                                                                 \
    cudaError_t _e = (EXPR);                                                           \
    TORCH_CHECK(_e == cudaSuccess, NAME, ": CUDA error: ", cudaGetErrorString(_e));    \
  } while (0)

inline void check_fp16_d64_aligned(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                   const char* name) {
  TORCH_CHECK(q.scalar_type() == at::kHalf, name, ": float16 only");
  TORCH_CHECK(q.size(3) == 64, name, ": only head dimension 64 is compiled (got ", q.size(3), ")");
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, name, ": inputs must be 16-byte aligned");
}

inline int smem_optin_bytes() {
  int dev = 0, optin = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  return optin;
}
