from typing import Tuple
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

sources = [str(Path(__file__).parent  / filename)
            for filename in 
          ["full_cumsum.cu", "segmented_sort_pairs.cu", "module.cpp"]]

cuda_lib = load("cuda_lib", sources=sources, verbose=True)


def full_cumsum(x:torch.Tensor) -> Tuple[torch.Tensor, int]:
  out = x.new_empty((x.shape[0] + 1, *x.shape[1:]))
  total = cuda_lib.full_cumsum(x, out)
  return out, total

segmented_sort_pairs = cuda_lib.segmented_sort_pairs

__all__ = ["full_cumsum", "segmented_sort_pairs"]


if __name__ == "__main__":
  k = torch.randint(100, (20,), dtype=torch.int32, device="cuda")
  v = torch.arange(20, dtype=torch.int32, device="cuda")

  segments = torch.tensor([[0, 8, 16], [8, 16, 20]], dtype=torch.long, device="cuda")

  k1, v1 = segmented_sort_pairs(k, v, segments)
  print(k1)
  print(v1)

  # y = full_cumsum(x)
  # print(y)

  # print(torch.cumsum(x, dim=0))
