import torch
import types

from dataclasses import dataclass
from typing import Optional, Tuple

from taichi_splatting.optim.fractional import FractionalOpt
from taichi_splatting.optim_slang import fractional_adam


class FractionalAdam(FractionalOpt):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    super().__init__(fractional_adam, params, lr, betas, eps, bias_correction)

class SparseAdam(FractionalOpt):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    super().__init__(fractional_adam, params, lr, betas, eps, bias_correction)

  def step(self, indexes: torch.Tensor, basis: Optional[torch.Tensor]=None):
    weight = torch.ones(indexes.shape[0], device=indexes.device, dtype=torch.float32)
    super().step(indexes, weight, basis)