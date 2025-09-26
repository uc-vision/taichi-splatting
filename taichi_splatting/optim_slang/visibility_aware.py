
import torch
import types

from typing import Optional
from beartype import beartype
from dataclasses import replace

from taichi_splatting.optim import VisibilityOptimizer
from taichi_splatting.optim_slang import fractional_adam, fractional_laprop

class VisibilityAwareAdam(VisibilityOptimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, vis_beta=0.5, 
               vis_smooth:float = 0.01, bias_correction=True, grad_clip:Optional[float]=None):
    super().__init__(fractional_adam, params,
                     lr=lr, betas=betas, eps=eps, vis_beta=vis_beta, 
                     vis_smooth=vis_smooth, bias_correction=bias_correction, grad_clip=grad_clip)

class VisibilityAwareLaProp(VisibilityOptimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, vis_beta=0.5, 
               vis_smooth:float=0.01, bias_correction=True, grad_clip:Optional[float]=None):
    
    
    super().__init__(fractional_laprop, params,
                     lr=lr, betas=betas, eps=eps, vis_beta=vis_beta, 
                     vis_smooth=vis_smooth, bias_correction=bias_correction, grad_clip=grad_clip)