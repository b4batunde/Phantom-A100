// PyTorch entry point for v3.1 / v3.2 (kernel: kernels/fused_tuned.cuh).
#include "common.cuh"
#include "kernels/fused_tuned.cuh"

torch::Tensor fused_tuned_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                                  bool double_buffer, int64_t br, int64_t bc) {
  check_fp16_d64_aligned(q, k, v, "fused_tuned");
  const bool compiled = (br == 64 || br == 128) && (bc == 32 || bc == 64);
  TORCH_CHECK(compiled, "fused_tuned: (br, bc) = (", br, ", ", bc,
              ") not compiled; available: (64,32) (64,64) (128,32) (128,64)");
  const size_t smem = attn::fused_tuned_smem_bytes((int)br, (int)bc, double_buffer);
  const int optin = smem_optin_bytes();
  TORCH_CHECK(smem <= (size_t)optin, "fused_tuned: BR=", br, " BC=", bc, " needs ", smem,
              " bytes of shared memory; this GPU allows ", optin);
  const int B = q.size(0), H = q.size(1), N = q.size(2);
  auto o = torch::empty_like(q);
  ATTN_CHECK(attn::launch_fused_tuned(dptr<__half>(q), dptr<__half>(k), dptr<__half>(v), dptr<__half>(o), B * H, N,
                                      causal, double_buffer, (int)br, (int)bc, at::cuda::getCurrentCUDAStream()),
             "fused_tuned");
  return o;
}
