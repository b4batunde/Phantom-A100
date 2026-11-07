// PyTorch entry point for v2 (kernels: kernels/tiled.cuh).
#include "common.cuh"
#include "kernels/tiled.cuh"

torch::Tensor tiled_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  auto S = torch::empty({B, H, N, N}, q.options().dtype(torch::kFloat32));
  auto o = torch::empty_like(q);
  DISPATCH_FLOAT_HALF(q.scalar_type(), "tiled_forward", [&] {
    ATTN_CHECK(attn::launch_tiled<T>(dptr<T>(q), dptr<T>(k), dptr<T>(v), dptr<T>(o), dptr<float>(S), B * H, N, d,
                                     causal, at::cuda::getCurrentCUDAStream()), "tiled_forward");
  });
  return o;
}
