#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cub/cub.cuh>


template <typename K, typename V>
void sort_helper(
  K *d_keys, V *d_values, 
  K *d_keys_out, V *d_values_out,
  int num_items,
  int begin_bit=0, int end_bit=-1) 
{
  size_t   temp_storage_bytes = 0;
  end_bit = end_bit > 0 ? end_bit : sizeof(K) * 8;
  auto stream = at::cuda::getCurrentCUDAStream();

  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out, num_items, begin_bit, end_bit, stream);

  auto temp_storage = torch::empty({int64_t(temp_storage_bytes)}, 
    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  cub::DeviceRadixSort::SortPairs(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out, num_items, begin_bit, end_bit, stream);

  cudaDeviceSynchronize();

}

template<typename K>
std::pair<torch::Tensor, torch::Tensor> sort_dispatch(const torch::Tensor keys, const torch::Tensor values, int begin_bit=0, int end_bit=-1) {
  auto keys_out = torch::empty_like(keys);
  auto values_out = torch::empty_like(values);
    
    sort_helper<K, int32_t>(
      keys.data_ptr<K>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<K>(), values_out.data_ptr<int32_t>(),
      keys.size(0), begin_bit, end_bit);

    return std::make_pair(keys_out, values_out);
}

std::pair<torch::Tensor, torch::Tensor> radix_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values, int begin_bit=0, int end_bit=-1) {
  
  assert (values.scalar_type() == torch::kI32), "values must be int32";
  assert (keys.dim() == 1 && values.dim() == 1), "keys and values must be 1D";
  assert (keys.size(0) == values.size(0)), "keys and values must have the same size";
  

  if (keys.scalar_type() == torch::kUInt16) {
    return sort_dispatch<uint16_t>(keys, values, begin_bit, end_bit);
  } else if (keys.scalar_type() == torch::kUInt32) {
    return sort_dispatch<uint32_t>(keys, values, begin_bit, end_bit);
  } else if (keys.scalar_type() == torch::kUInt64) {
    return sort_dispatch<uint64_t>(keys, values, begin_bit, end_bit);
  } else if (keys.scalar_type() == torch::kI16) {
    return sort_dispatch<int16_t>(keys, values, begin_bit, end_bit);
  } else if (keys.scalar_type() == torch::kI32) {
    return sort_dispatch<int32_t>(keys, values, begin_bit, end_bit);
  } else if (keys.scalar_type() == torch::kI64) {
    return sort_dispatch<int64_t>(keys, values, begin_bit, end_bit);

  } else { 
      throw std::runtime_error("Not yet implemented for data type(s).");
  }

  
}

