import torch

def check_finite(tensor_dict):
  for k, v in tensor_dict.items():
    n = (~torch.isfinite(v)).sum()
    if n > 0:
      raise ValueError(f'Found {n} non-finite values in {k}')

    if v.grad is not None:
      n = (~torch.isfinite(v.grad)).sum()
      if n > 0:
        raise ValueError(f'Found {n} non-finite gradients in {k}')