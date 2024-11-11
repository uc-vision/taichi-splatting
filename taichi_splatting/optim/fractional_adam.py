from functools import cache
import torch
import taichi as ti
from typing import Optional

from taichi_splatting.taichi_queue import queued

@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)


@cache
def fractional_adam_kernel(betas=(0.9, 0.999), eps=1e-16):
  beta1, beta2 = betas

  @queued
  @ti.kernel
  def kernel(param_step: ti.types.ndarray(dtype=ti.f32, ndim=2),   # M x D - Output step (to be scaled by lr)
             
             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1),    # M Visible indexes
             weight: ti.types.ndarray(dtype=ti.f32, ndim=1),       # M weight of each visible index

             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),      # N x D - Running average of gradient
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2),   # N x D - Running average of gradient squared
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1), # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),          # N x D Gradient input

             lr: ti.f32, # Learning rate
          ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]

      b1 = beta1 ** w 
      b2 = beta2 ** w

      bias_factor = ti.sqrt(1 - b2 ** total_weight[idx])  / (1 - b1 ** total_weight[idx])
      for j in range(param_step.shape[1]):
        g = grad[idx, j]

        avg = lerp(b1, exp_avg[idx, j], g)
        avg_sq = lerp(b2, exp_avg_sq[idx, j], g * g)

        param_step[i, j] = (avg / ti.max(ti.sqrt(avg_sq),  eps)) * bias_factor * w * lr

        exp_avg[idx, j] = avg
        exp_avg_sq[idx, j] = avg_sq


  return kernel

@cache
def fractional_vector_adam_kernel(betas=(0.9, 0.999), eps=1e-16, dims=3):
  b1, b2 = betas
  vec = ti.types.vector(n=dims, dtype=ti.f32)

  @queued
  @ti.kernel
  def kernel(param_step: ti.types.ndarray(dtype=vec, ndim=1), # M x D - Output step (to be scaled by lr)
             
             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1),     # M visible indexes
             weight: ti.types.ndarray(dtype=ti.f32, ndim=1),        # M weight of each visible index

             exp_avg: ti.types.ndarray(dtype=vec, ndim=1),          # N x D - Running average of gradient
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1),    # N  - Running norm of gradient 
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1),  # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=vec, ndim=1),             # N x D - Gradient input

             lr: ti.f32, # Learning rate
        ):

    for i in indexes:
      idx = indexes[i]
      bias_factor = ti.sqrt(1 - b2 ** total_weight[idx])  / (1 - b1 ** total_weight[idx])

      g = grad[idx]
      avg = lerp(b1, exp_avg[idx], g)

      norm = ti.math.dot(g, g)
      avg_sq = lerp(b2, exp_avg_sq[idx], norm)

      param_step[i] = (avg / ti.max(ti.sqrt(avg_sq),  eps)) * bias_factor * weight[i] * lr

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel




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
    state['exp_avg'] = torch.zeros_like(param.view(param.shape[0], -1))
    state['exp_avg_sq'] = torch.zeros((param.shape[0],), dtype=param.dtype, device=param.device)

  return state


def get_scalar_state(state:dict, param: torch.Tensor):
  if len(state) == 0:
    state['exp_avg'] = torch.zeros_like(param.view(param.shape[0], -1))
    state['exp_avg_sq'] = torch.zeros_like(param.view(param.shape[0], -1))

  return state




def scalar_adam_step(group:dict, param: torch.Tensor, state: dict,
    total_weight: torch.Tensor,
    visible_weight: torch.Tensor,
    visible_indexes: torch.Tensor,
  ):

  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 
  param_step = param.new_zeros(visible_indexes.shape[0], param.shape[1])

  kernel = fractional_adam_kernel(betas=group["betas"], eps=group["eps"])
  
  kernel(param_step, 
         visible_indexes, visible_weight,
         state['exp_avg'], state['exp_avg_sq'], total_weight,
         grad, group["lr"])
  
  return param_step


def vector_adam_step(group:dict, param: torch.Tensor, state: dict, 
      total_weight: torch.Tensor,
      visible_weight: torch.Tensor,
      visible_indexes: torch.Tensor, 
      basis: Optional[torch.Tensor]=None
  ): 
  assert basis is None or group["type"] == "local_vector", "basis is required for local_vector optimizer"

  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 
  param_step = param.new_zeros(visible_indexes.shape[0], param.shape[1])

  dim = param.shape[1]

  kernel = fractional_vector_adam_kernel(betas=group["betas"], eps=group["eps"], dims=dim)

  if basis is not None:
    inv_basis = torch.linalg.inv(basis)
    grad[visible_indexes] = torch.einsum('bij,bj->bi', inv_basis, grad[visible_indexes])

  kernel(param_step, 
         visible_indexes, visible_weight, 
         state['exp_avg'], state['exp_avg_sq'], total_weight,
         grad, group["lr"])

  if basis is not None:
    param_step = torch.einsum('bij,bj->bi', basis, param_step)

  return param_step


@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))


class FractionalAdam(torch.optim.Optimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), vis_beta=0.9, eps=1e-16):
    
    if not 0.0 <= lr:
      raise ValueError(f"Invalid learning rate: {lr}")
    
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta1 {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta2 {betas[1]}")
    if not 0.0 <= vis_beta < 1.0:
      raise ValueError(f"Invalid visibility beta {vis_beta}")
    

    self.vis_beta = vis_beta

    defaults = dict(lr=lr, betas=betas, eps=eps, point_lr=None, mask_lr=None, basis=None, type="scalar")  
    self.shared_state = {}

    super().__init__(params, defaults)

  def update_group(self, param, **kwargs):
    for group in self.param_groups:
      if param in group["params"]:
        group.update(kwargs)


  def update_visibility(self, visibility: torch.Tensor, visible_indexes: torch.Tensor, n:int):
    if 'weight' not in self.shared_state:
      self.shared_state['total_weight'] = torch.zeros(n, dtype=torch.float32, device=visibility.device)
      self.shared_state['running_vis'] = torch.zeros(n, dtype=torch.float32, device=visibility.device)

    total_weight, running_vis = self.shared_state['total_weight'], self.shared_state['running_vis']

    updated_vis = exp_lerp(self.vis_beta, running_vis[visible_indexes], visibility)
    weight = visibility / updated_vis

    running_vis[visible_indexes] = updated_vis
    total_weight.index_add_(0, visible_indexes, weight)

    return weight, total_weight


  @torch.no_grad()
  def step(self, 
           visible_indexes: torch.Tensor, 
           visibility: torch.Tensor,
           basis: Optional[torch.Tensor]=None):
    
    named = {group["name"]: group for group in self.param_groups}
    keys = list(named.keys())
    
    def get_params(k):
      group = named[k]

      n = len(group["params"])
      assert n == 1, f"expected 1 tensor in group {k}, got {n}"
      return group["params"][0]
    
    assert visibility.shape == visible_indexes.shape, f"shape mismatch {visibility.shape} != {visible_indexes.shape}"

    n = get_params(keys[0]).shape[0]
    weight, total_weight = self.update_visibility(visibility, visible_indexes, n)


    for k, group in named.items():
      param = get_params(k)
      if param.grad is None:
        continue
      
      assert param.shape[0] == n, f"param shape {param.shape[0]} != {n}"

      state = self.state[param]
      group_type = group["type"]

      if group_type in ["vector", "local_vector"]:
        param_step = vector_adam_step(group, param, 
                  get_vector_state(state, param), 

                  total_weight=total_weight,
                  visible_weight=weight, 
                  visible_indexes=visible_indexes, 
                  basis=basis if group_type == "local_vector" else None)
        
      elif group_type == "scalar":
        param_step = scalar_adam_step(group, param, 
                  get_scalar_state(state, param), 

                  total_weight=total_weight,
                  visible_weight=weight, 
                  visible_indexes=visible_indexes)
      else:
        raise ValueError(f"unknown group type {group_type}")
    
      print(param_step.shape)


      # Apply per-point and per-element learning rates
      if group["point_lr"] is not None:
        param_step *= group["point_lr"][visible_indexes].unsqueeze(1)
      if group["mask_lr"] is not None:
        param_step *= group["mask_lr"].unsqueeze(0)


      param = param.view(param.shape[0], -1) 
      param[visible_indexes] -= param_step

