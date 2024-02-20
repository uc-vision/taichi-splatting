from beartype.typing import Mapping
import torch

def check_finite(name, x):

  if isinstance(x, torch.Tensor):
    n = (~torch.isfinite(x)).sum()
    if n > 0:
      raise ValueError(f'Found {n} non-finite values in tensor {name}')
    
    if x.grad is not None:
      check_finite(f'{name}.grad', x.grad)

  elif isinstance(x, Mapping):
    for k, v in x.items():
      check_finite(f'{name}.{k}', v)

  else:
    raise AssertionError(f'Unsupported type {type(x)}')


