import torch
from typing import Mapping, Sequence
from tensordict.tensorclass import TensorClass

def check_finite(t, name, warn=False):

  d = count_nonfinite(t, name, warn)
  if len(d) > 0:
    if warn:
      print(f'Non-finite entries: {d}')
    else:
      raise ValueError(f'Non-finite entries: {d}')
  

def append_nonfinite(d: dict, name: str, t: torch.Tensor):
  n = (~torch.isfinite(t)).sum()
  if n > 0:
    d[name] = n

  return d

def count_nonfinite(t, name, warn=False) -> dict:
  d = {}
  if isinstance(t, torch.Tensor):
    append_nonfinite(d, name, t)
    if t.grad is not None:
      append_nonfinite(d, f'{name}.grad', t.grad)
    return d
  
  if isinstance(t, Sequence) and not isinstance(t, str):
    for i, v in enumerate(t):
      d.update(count_nonfinite(v, f'{name}[{i}]', warn))
    return d
  
  if isinstance(t, Mapping):
    for k, v in t.items():
      d.update(count_nonfinite(v, f'{name}.{k}', warn))
    return d

  if isinstance(t, TensorClass):
    for k, v in t.items():
      d.update(count_nonfinite(v, f'{name}.{k}', warn))
    return d

  return {}
