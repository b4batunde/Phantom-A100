// PyTorch entry point for v4 (kernel: kernels/fused_tc.cuh).
#include "common.cuh"
#include "kernels/fused_tc.cuh"

torch::Tensor fused_tc_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  check_fp16_d64_aligned(q, k, v, "fused_tc");
  int dev = 0, major = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  TORCH_CHECK(major >= 7, "fused_tc: tensor cores need compute capability 7.0 or newer");
  const int optin = smem_optin_bytes();
  TORCH_CHECK(attn::fused_tc_smem_bytes() <= (size_t)optin, "fused_tc: needs ", attn::fused_tc_smem_bytes(),
              " bytes of shared memory, GPU allows ", optin);
  const int B = q.size(0), H = q.size(1), N = q.size(2);
  auto o = torch::empty_like(q);
  ATTN_CHECK(attn::launch_fused_tc(dptr<__half>(q), dptr<__half>(k), dptr<__half>(v), dptr<__half>(o), B * H, N,
                                   causal, at::cuda::getCurrentCUDAStream()), "fused_tc");
  return o;
}
