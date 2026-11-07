// PyTorch entry point for v1 (kernels: kernels/naive.cuh).
#include "common.cuh"
#include "kernels/naive.cuh"

torch::Tensor naive_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  auto S = torch::empty({B, H, N, N}, q.options().dtype(torch::kFloat32));
  auto o = torch::empty_like(q);
  DISPATCH_FLOAT_HALF(q.scalar_type(), "naive_forward", [&] {
    ATTN_CHECK(attn::launch_naive<T>(dptr<T>(q), dptr<T>(k), dptr<T>(v), dptr<T>(o), dptr<float>(S), B * H, N, d,
                                     causal, at::cuda::getCurrentCUDAStream()), "naive_forward");
  });
  return o;
}
