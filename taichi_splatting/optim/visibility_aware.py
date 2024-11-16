from dataclasses import replace
import types
from typing import Optional
import torch

from taichi_splatting.optim import fractional_adam, fractional_laprop
from taichi_splatting.optim.fractional import make_group, weighted_step
from taichi_splatting.optim.util import get_total_weight

def get_running_vis(state:dict,  n:float, device:torch.device):
  if 'running_vis' not in state:
    state['running_vis'] = torch.zeros( (n,), device=device, dtype=torch.float32)

  return state['running_vis']

@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))


def update_visibility(running_vis: torch.Tensor, 
                      visibility: torch.Tensor, visible_indexes: torch.Tensor, 
                      beta: float = 0.9):

  updated_vis = exp_lerp(beta, running_vis[visible_indexes], visibility)
  weight = visibility / (updated_vis + 1e-8)
  running_vis[visible_indexes] = updated_vis
  return weight


def set_indexes(target:torch.Tensor, values:torch.Tensor, indexes:torch.Tensor):
  result = torch.zeros_like(target)
  result[indexes] = values
  return result


class VisibilityOptimizer(torch.optim.Optimizer):
  
  def __init__(self, kernels:types.ModuleType, param_groups:list[dict], lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16):
    
    assert lr > 0, f"Invalid learning rate: {lr}"
    assert eps > 0, f"Invalid epsilon: {eps}"
    assert 0.0 <= betas[0] < 1.0, f"Invalid beta1: {betas[0]}"
    assert 0.0 <= betas[1] < 1.0, f"Invalid beta2: {betas[1]}"

    defaults = dict(lr=lr, betas=betas, eps=eps, mask_lr=None, type="scalar")  

    self.kernels = kernels
    super().__init__(param_groups, defaults)


  @torch.no_grad()
  def step(self, 
          visible_indexes: torch.Tensor, 
          visibility: torch.Tensor, 
          basis: Optional[torch.Tensor]=None):
    
    assert visibility.shape == visible_indexes.shape, f"shape mismatch {visibility.shape} != {visible_indexes.shape}"

    groups = [make_group(group, self.state) for group in self.param_groups]
    n = groups[0].num_points

    total_weight = get_total_weight(groups[0].state, n, device=visibility.device)
    running_vis = get_running_vis(groups[0].state, n, device=visibility.device)

    weight = update_visibility(running_vis, visibility, visible_indexes)
    total_weight[visible_indexes] += weight

    for group in groups:
      if group.grad is None: 
        continue

      assert group.num_points == n, f"param shape {group.num_points} != {n}"
      group = replace(group, grad=set_indexes(group.grad, 
                        group.grad[visible_indexes] / visibility.unsqueeze(1), 
                        visible_indexes))

      lr_step = weighted_step(group, weight, visible_indexes, total_weight, self.kernels, basis)
      group.param[visible_indexes] -= lr_step * weight.sqrt().unsqueeze(1)


class VisibilityAwareAdam(VisibilityOptimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16):
    super().__init__(fractional_adam, params, lr, betas, eps)


class VisibilityAwareLaProp(VisibilityOptimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16):
    super().__init__(fractional_laprop, params, lr, betas, eps)