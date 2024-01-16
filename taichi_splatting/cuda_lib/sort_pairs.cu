#include <torch/extension.h>
#include <cub/cub.cuh>

template <typename K, typename V>
void sort_helper(
  K *d_keys, V *d_values, 
  K *d_keys_out, V *d_values_out,

  int num_items) 
{
  size_t   temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out, num_items);

  auto temp_storage = torch::empty({int64_t(temp_storage_bytes)}, 
    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  cub::DeviceRadixSort::SortPairs(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out, num_items);

}

std::pair<torch::Tensor, torch::Tensor> sort_pairs(
  const torch::Tensor keys, const torch::Tensor values) {
  
  assert (keys.dim() == 1 && values.dim() == 1), "keys and values must be 1D";
  assert (keys.size(0) == values.size(0)), "keys and values must have the same size";
  
  auto keys_out = torch::empty_like(keys);
  auto values_out = torch::empty_like(values);

  if (keys.scalar_type() == torch::kInt32 && values.scalar_type() == torch::kInt32) {
    sort_helper<int32_t, int32_t>(
      keys.data_ptr<int32_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int32_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0));
      return std::make_pair(keys_out, values_out);
  } else if (keys.scalar_type() == torch::kInt64 && values.scalar_type() == torch::kInt32) {
    sort_helper<int64_t, int32_t>(
      keys.data_ptr<int64_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int64_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0));
      return std::make_pair(keys_out, values_out);

  } else { 
      // TODO, add all the other cases.
      throw std::runtime_error("Not yet implemented for data type(s).");
  }
}

