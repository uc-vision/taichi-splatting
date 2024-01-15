from typing import Tuple
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

path = Path(__file__).parent  / "scan.cu"

cuda_lib = load("cuda_lib", sources=[str(path)], verbose=True)


def full_cumsum(x:torch.Tensor) -> Tuple[torch.Tensor, int]:
  out = x.new_empty((x.shape[0] + 1, *x.shape[1:]))
  total = cuda_lib.full_cumsum(x, out)
  return out, total


__all__ = ["full_cumsum"]


if __name__ == "__main__":
  x = torch.randint(100, (10,), dtype=torch.int32, device="cuda")
  y = full_cumsum(x)
  print(y)

  print(torch.cumsum(x, dim=0))
