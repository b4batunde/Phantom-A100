#pragma once
// Pure CUDA helpers shared by every kernel. No PyTorch headers here, so the same kernels can be
// built into the PyTorch extension AND into the standalone simulator harness (sim/harness.cu).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

namespace attn {

// Load/store helpers so every kernel can be templated on float or half.
// All math is done in FP32 regardless of the storage type.
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr float kLog2e = 1.4426950408889634f;

}  // namespace attn
