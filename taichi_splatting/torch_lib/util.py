from beartype import beartype
import torch
from typing import Mapping, Sequence
from tensordict.tensorclass import is_tensorclass

@beartype
def check_finite(t, name:str, warn=False):

  d = count_nonfinite(t, name)
  if len(d) == 0:
    raise ValueError(f'No tensors found in {name}, type {type(t)} to check for non-finite entries')

  non_finite = {k: v for k, v in d.items() if v > 0}
  if len(non_finite) > 0:
    if warn:
      print(f'Non-finite entries: {non_finite}')
    else:
      raise ValueError(f'Non-finite entries: {non_finite}')
  

def append_nonfinite(d: dict, name: str, t: torch.Tensor):
  d[name] = (~torch.isfinite(t)).sum().item()
  return d

@beartype
def count_nonfinite(t, name:str) -> dict:
  d = {}
  
  if isinstance(t, torch.Tensor):
    append_nonfinite(d, name, t)
    if t.grad is not None:
      append_nonfinite(d, f'{name}.grad', t.grad)
    return d
  
  if isinstance(t, Sequence) and not isinstance(t, str):
    for i, v in enumerate(t):
      d.update(count_nonfinite(v, f'{name}[{i}]'))
    return d
  
  if isinstance(t, Mapping):
    for k, v in t.items():
      d.update(count_nonfinite(v, f'{name}.{k}'))
    return d

  if is_tensorclass(t):
    for k, v in t.items():
      d.update(count_nonfinite(v, f'{name}.{k}'))
    return d


  return {}
