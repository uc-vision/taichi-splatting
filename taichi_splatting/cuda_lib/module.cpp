#include <pybind11/pybind11.h>
#include <torch/extension.h>


int64_t full_cumsum(const torch::Tensor input, const torch::Tensor output);
std::pair<torch::Tensor, torch::Tensor> segmented_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values,
  const torch::Tensor segment_ranges);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("full_cumsum", &full_cumsum, "Full cumulative sum");
  m.def("segmented_sort_pairs", &segmented_sort_pairs, "Segmented sort pairs");
}