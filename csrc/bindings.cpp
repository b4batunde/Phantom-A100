#include <torch/extension.h>

torch::Tensor naive_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor tiled_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor fused_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor fused_tuned_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                                  bool double_buffer, int64_t br, int64_t bc);
torch::Tensor fused_tc_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);

namespace {

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  for (const auto* t : {&q, &k, &v}) {
    TORCH_CHECK(t->is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(t->is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(t->dim() == 4, "inputs must be shaped (B, H, N, d)");
    TORCH_CHECK(t->scalar_type() == q.scalar_type(), "q, k, v must share a dtype");
  }
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v must have the same shape");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on the same device");
  TORCH_CHECK(q.numel() > 0, "inputs must be non-empty");
}

using Fn = torch::Tensor (*)(torch::Tensor, torch::Tensor, torch::Tensor, bool);

template <Fn F>
torch::Tensor checked(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  check_inputs(q, k, v);
  return F(q, k, v, causal);
}

torch::Tensor checked_tuned(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                            bool double_buffer, int64_t br, int64_t bc) {
  check_inputs(q, k, v);
  return fused_tuned_forward(q, k, v, causal, double_buffer, br, bc);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Hand-written CUDA attention kernels: naive, tiled, fused, fused_tuned, fused_tc";
  m.def("naive", &checked<naive_forward>, "v1 naive attention forward",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
  m.def("tiled", &checked<tiled_forward>, "v2 shared-memory tiled attention forward",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
  m.def("fused", &checked<fused_forward>, "v3 fused online-softmax attention forward",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
  m.def("fused_tuned", &checked_tuned,
        "v3.1/v3.2 fused attention with 128-bit loads and optional cp.async double buffering (fp16)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
        py::arg("double_buffer") = true, py::arg("br") = 64, py::arg("bc") = 32);
  m.def("fused_tc", &checked<fused_tc_forward>, "v4 fused attention on tensor cores via WMMA (fp16)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
