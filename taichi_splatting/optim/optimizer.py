from functools import cache
import torch
import taichi as ti
from typing import Callable, Optional

from taichi_splatting.optim.fractional_adam import fractional_adam_kernel, fractional_vector_adam_kernel
# from taichi_splatting.optim.fractional_laprop import fractional_vector_laprop_kernel
from taichi_splatting.taichi_queue import queued


def get_point_lr(group: dict, param: torch.Tensor):
  if group["point_lr"] is not None:
    point_lr=group["point_lr"]
    assert point_lr.shape == (param.shape[0],), f"point_lr shape {point_lr.shape} != {param.shape[0]}"

    return point_lr, True
  else:
    return torch.empty((0,), device=param.device, dtype=torch.float32), False

def get_mask_lr(group: dict, param: torch.Tensor):
  if group["mask_lr"] is not None:
    mask_lr = group["mask_lr"].view(-1)
    assert mask_lr.shape == param.shape[1:], f"mask_lr shape {mask_lr.shape} != {param.shape[1:]}"
    return mask_lr, True
  else:
    return torch.empty((0,), device=param.device, dtype=torch.float32), False


def get_vector_state(state:dict, param: torch.Tensor):
  if len(state) == 0:
    state['step'] = torch.zeros((param.shape[0],), dtype=param.dtype, device=param.device)
    state['exp_avg'] = torch.zeros_like(param.view(param.shape[0], -1))
    state['exp_avg_sq'] = torch.zeros((param.shape[0],), dtype=param.dtype, device=param.device)

  return state


def get_scalar_state(state:dict, param: torch.Tensor):
  if len(state) == 0:
    state['step'] = torch.zeros((param.shape[0],), dtype=param.dtype, device=param.device)
    state['exp_avg'] = torch.zeros_like(param.view(param.shape[0], -1))
    state['exp_avg_sq'] = torch.zeros_like(param.view(param.shape[0], -1))

  return state



def scalar_step(
    group:dict, param: torch.Tensor, state: dict,
    visible_weight: torch.Tensor,
    visible_indexes: torch.Tensor,
    make_kernel:Callable = fractional_adam_kernel,
  ):

  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 
  lr_step = param.new_zeros(visible_indexes.shape[0], param.shape[1])

  kernel = make_kernel(betas=group["betas"], eps=group["eps"])
  kernel(lr_step, 
         visible_indexes, visible_weight,
         state['exp_avg'], state['exp_avg_sq'], state['total_weight'],
         grad, group["lr"])
  
  return lr_step


def vector_step(
    group:dict, param: torch.Tensor, state: dict, 
    visible_weight: torch.Tensor,
    visible_indexes: torch.Tensor, 
    basis: Optional[torch.Tensor]=None,
    make_kernel:Callable = fractional_vector_adam_kernel,
  ): 
  assert basis is None or group["type"] == "local_vector", "basis is required for local_vector optimizer"

  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 
  lr_step = param.new_zeros(visible_indexes.shape[0], param.shape[1])

  dim = param.shape[1]
  kernel = make_kernel(betas=group["betas"], eps=group["eps"], dims=dim)

  if basis is not None:
    inv_basis = torch.linalg.inv(basis)
    grad[visible_indexes] = torch.einsum('bij,bj->bi', inv_basis, grad[visible_indexes])

  kernel(lr_step, 
         visible_indexes, visible_weight, 
         state['exp_avg'], state['exp_avg_sq'], state['total_weight'],
         grad, group["lr"])

  if basis is not None:
    lr_step = torch.einsum('bij,bj->bi', basis, lr_step)

  return lr_step


@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))


def update_visibility(running_vis: torch.Tensor, 
                      visibility: torch.Tensor, visible_indexes: torch.Tensor, 
                      beta: float = 0.9):

  updated_vis = exp_lerp(beta, running_vis[visible_indexes], visibility)
  weight = visibility / updated_vis
  running_vis[visible_indexes] = updated_vis
  return weight, running_vis



class FractionalAdam(torch.optim.Optimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16):
    
    assert lr > 0, f"Invalid learning rate: {lr}"
    assert eps > 0, f"Invalid epsilon: {eps}"
    assert 0.0 <= betas[0] < 1.0, f"Invalid beta1: {betas[0]}"
    assert 0.0 <= betas[1] < 1.0, f"Invalid beta2: {betas[1]}"

    defaults = dict(lr=lr, betas=betas, eps=eps, point_lr=None, mask_lr=None, basis=None, type="scalar")  
    self.shared_state = {}

    super().__init__(params, defaults)

  def update_group(self, param, **kwargs):
    for group in self.param_groups:
      if param in group["params"]:
        group.update(kwargs)

  def check_group(self, group):
    n = len(group["params"])  
    assert n == 1, f"expected 1 tensor in group {group['name']}, got {n}"
    params = group["params"][0]
    return (params, group, self.state[params])
    

  @torch.no_grad()
  def step(self, 
           visible_indexes: torch.Tensor, 
           weight: torch.Tensor, 
           basis: Optional[torch.Tensor]=None):
    
    assert weight.shape == visible_indexes.shape, f"shape mismatch {weight.shape} != {visible_indexes.shape}"

    params = [self.check_group(group) for group in self.param_groups]
    n = params[0][0].shape[0]

    for (param, group, state) in params:
      if param.grad is None:
        continue
      
      assert param.shape[0] == n, f"param shape {param.shape[0]} != {n}"
      group_type = group["type"]

      if group_type in ["vector", "local_vector"]:
        lr_step = vector_step(group, param, 
                  get_vector_state(state, param), 
                  visible_weight=weight, 
                  visible_indexes=visible_indexes, 
                  basis=basis if group_type == "local_vector" else None)
        
      elif group_type == "scalar":
        lr_step = scalar_step(group, param, 
                  get_scalar_state(state, param), 
                  visible_weight=weight, 
                  visible_indexes=visible_indexes)
      else:
        raise ValueError(f"unknown group type {group_type}")
    

      # Apply per-point and per-element learning rates
      if group["point_lr"] is not None:
        lr_step *= group["point_lr"][visible_indexes].unsqueeze(1)
      if group["mask_lr"] is not None:
        lr_step *= group["mask_lr"].unsqueeze(0)


      param = param.view(param.shape[0], -1) 
      param[visible_indexes] -= lr_step

