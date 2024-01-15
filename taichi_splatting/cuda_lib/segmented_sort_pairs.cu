#include <torch/extension.h>
#include <cub/cub.cuh>

template <typename K, typename V>
void sort_helper(
  K *keys, V *values, 
  K *keys_out, V *values_out,

  int num_items, 
  int64_t *start_offset, int64_t *end_offset, 
  int num_segments) 
{

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      keys, keys_out, values, values_out,
      num_items, num_segments, start_offset, end_offset);

  auto temp_storage_tensor_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto temp_storage_tensor = torch::empty({int64_t(temp_storage_bytes)}, temp_storage_tensor_options);
  assert(temp_storage_tensor.nbytes() >= temp_storage_bytes);
  d_temp_storage = temp_storage_tensor.data_ptr<uint8_t>();

  cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      keys, keys_out, values, values_out,
      num_items, num_segments, start_offset, end_offset);

}


std::pair<torch::Tensor, torch::Tensor> segmented_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values,
  const torch::Tensor segment_ranges) {
  
  assert (keys.dim() == 1 && values.dim() == 1), "keys and values must be 1D";
  assert (keys.size(0) == values.size(0)), "keys and values must have the same size";

  assert (segment_ranges.dim() == 2 && segment_ranges.size(0) == 2), 
    "segment_ranges must be 2D start, end pairs 2,N"; 

  assert (segment_ranges.scalar_type() == torch::kInt64), "segment_ranges must be int64";
  
  auto keys_out = torch::empty_like(keys);
  auto values_out = torch::empty_like(values);

  if (keys.scalar_type() == torch::kInt32 && values.scalar_type() == torch::kInt32) {
    sort_helper<int32_t, int32_t>(
      keys.data_ptr<int32_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int32_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), 
      segment_ranges[0].data_ptr<int64_t>(), segment_ranges[1].data_ptr<int64_t>(),
      segment_ranges.size(1));

      return std::make_pair(keys_out, values_out);
  } else { 
      // TODO, add all the other cases.
      throw std::runtime_error("Not yet implemented for data type.");
  }
}

