// PyTorch entry point for v3 (kernel: kernels/fused.cuh).
#include "common.cuh"
#include "kernels/fused.cuh"

torch::Tensor fused_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  TORCH_CHECK(d == 64, "fused_forward: only head dimension 64 is compiled (got ", d, ")");
  auto o = torch::empty_like(q);
  DISPATCH_FLOAT_HALF(q.scalar_type(), "fused_forward", [&] {
    ATTN_CHECK(attn::launch_fused<T>(dptr<T>(q), dptr<T>(k), dptr<T>(v), dptr<T>(o), B * H, N, causal,
                                     at::cuda::getCurrentCUDAStream()), "fused_forward");
  });
  return o;
}
