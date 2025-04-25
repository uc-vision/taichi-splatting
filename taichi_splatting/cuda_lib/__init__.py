from beartype.typing import Tuple
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

sources = [str(Path(__file__).parent  / filename)
            for filename in 
          ["full_cumsum.cu", "radix_sort_pairs.cu", "segmented_sort_pairs.cu", "module.cpp"]]

cuda_lib = load("taichi_splatting_cuda_lib", sources=sources, verbose=True)


def full_cumsum(x:torch.Tensor) -> Tuple[torch.Tensor, int]:
  assert x.dtype in [torch.int16, torch.int32, torch.int64], \
      f"full_cumsum: tensor must be integer type, got {x.dtype}"

  if x.shape[0] == 0:
    return x.new_zeros((1, )), 0
  
  if x.dtype == torch.int16:
    out = x.new_empty((x.shape[0] + 1, ), dtype=torch.int32)
  else:
    out = x.new_empty((x.shape[0] + 1, *x.shape[1:]))

  total = cuda_lib.full_cumsum(x, out)
  return out, total

segmented_sort_pairs = cuda_lib.segmented_sort_pairs
def radix_sort_pairs(keys:torch.Tensor, values:torch.Tensor, start_bit=0, end_bit=None, unsigned=False):
  if end_bit is None:
    end_bit = -1

  return cuda_lib.radix_sort_pairs(keys, values, start_bit, end_bit, unsigned)

__all__ = ["full_cumsum", "radix_sort_pairs", "segmented_sort_pairs"]


if __name__ == "__main__":
  k = torch.randint(100, (20,), dtype=torch.int32, device="cuda")
  v = torch.arange(20, dtype=torch.int32, device="cuda")

  start_offsets = torch.tensor([0, 8, 16], dtype=torch.long, device="cuda")
  end_offsets = torch.tensor([8, 16, 20], dtype=torch.long, device="cuda")

  k1, v1 = segmented_sort_pairs(k, v, start_offsets, end_offsets)
  print(k1)
  print(v1)

  k2, v2 = radix_sort_pairs(k.long(), v)
  print(k2, k2.dtype)
  print(v2)

  # y = full_cumsum(x)
  # print(y)

  # print(torch.cumsum(x, dim=0))
