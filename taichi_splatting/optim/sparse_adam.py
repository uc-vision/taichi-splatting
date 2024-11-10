from functools import cache
import torch
import taichi as ti
from typing import Optional

from taichi_splatting.taichi_queue import queued

@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)


@cache
def adam_kernel(betas=(0.9, 0.999), eps=1e-16,  use_point_lr=False, use_mask_lr=False):
  beta1, beta2 = betas


  @queued
  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),  # N x D


             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes
             weight: ti.types.ndarray(dtype=ti.f32, ndim=1),    # M weight of each visible index

             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1), # N step for each point (total weight)

             point_lr: ti.types.ndarray(dtype=ti.f32, ndim=1), # N learning rate multipliers across points
             mask_lr: ti.types.ndarray(dtype=ti.f32, ndim=1),  # D learning rate multipliers for each member of a param vector

             global_lr: ti.f32):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]

      b1 = beta1 ** w 
      b2 = beta2 ** w

      lr_idx = point_lr[idx] if ti.static(use_point_lr) else 1.0
      bias_factor = ti.sqrt(1 - b2 ** total_weight[idx])  / (1 - b1 ** total_weight[idx])

      for j in range(param.shape[1]):
        g = grad[idx, j]

        avg = lerp(b1, exp_avg[idx, j], g)
        avg_sq = lerp(b2, exp_avg_sq[idx, j], g * g)

        lr = global_lr * lr_idx * (mask_lr[j] if ti.static(use_mask_lr) else 1.0) 
        param[idx, j] -= lr * (avg / ti.max(ti.sqrt(avg_sq),  eps)) * bias_factor * ti.math.sqrt(w)

        exp_avg[idx, j] = avg
        exp_avg_sq[idx, j] = avg_sq


  return kernel

@cache
def vector_adam_kernel(betas=(0.9, 0.999), eps=1e-16, dims=3, use_point_lr=False):
  b1, b2 = betas
  vec = ti.types.vector(n=dims, dtype=ti.f32)


  @queued
  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=vec, ndim=1), # N x D
             grad: ti.types.ndarray(dtype=vec, ndim=1),  # N x D

             step: ti.types.ndarray(dtype=ti.f32, ndim=1), # N

             exp_avg: ti.types.ndarray(dtype=vec, ndim=1),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes
             point_lr: ti.types.ndarray(dtype=ti.f32, ndim=1), # N learning rate multipliers across points

             lr: ti.f32):

    for i in indexes:
      idx = indexes[i]
      bias_factor = ti.sqrt(1 - b2 ** step[idx])  / (1 - b1 ** step[idx])


      g = grad[idx]
      avg = lerp(b1, exp_avg[idx], g)

      norm = ti.math.dot(g, g)
      avg_sq = lerp(b2, exp_avg_sq[idx], norm)

      lr_idx = point_lr[idx] if ti.static(use_point_lr) else 1.0
      param[idx] -= lr * avg / ti.max(ti.sqrt(avg_sq),  eps) * bias_factor * lr_idx

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel


# @cache
# def local_vector_adam_kernel(basis_type, to_local, from_local, betas=(0.9, 0.999), eps=1e-16, dims=2):
#   b1, b2 = betas
#   vec = ti.types.vector(n=dims, dtype=ti.f32)
  
#   @queued
#   @ti.kernel
#   def kernel(param: ti.types.ndarray(dtype=vec, ndim=1), # N 
#              grad: ti.types.ndarray(dtype=vec, ndim=1),  # N 

#              step: ti.types.ndarray(dtype=ti.f32, ndim=1), # N

#              exp_avg: ti.types.ndarray(dtype=vec, ndim=1),    # N 
#              exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1), # N 

#              indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes
#              basis: ti.types.ndarray(dtype=basis_type, ndim=1), # N 

#              lr: ti.f32):

#     for i in indexes:
#       idx = indexes[i]

#       step[idx] += 1
#       bias_factor = ti.sqrt(1 - b2 ** step[idx])  / (1 - b1 ** step[idx])

#       local_grad = to_local(grad[idx], basis[i])
#       avg = lerp(b1, exp_avg[idx], local_grad)

#       norm = ti.math.dot(local_grad, local_grad)
#       avg_sq = lerp(b2, exp_avg_sq[idx], norm)

#       local_step = lr * avg / ti.max(ti.sqrt(avg_sq),  eps) * bias_factor
#       param[idx] -= from_local(local_step, basis[i])

#       exp_avg[idx] = avg
#       exp_avg_sq[idx] = avg_sq

#   return kernel

# @cache
# def basis_kernel(betas=(0.9, 0.999), eps=1e-16, dims=2):

#   vec = ti.types.vector(n=dims, dtype=ti.f32)
#   basis = ti.types.matrix(n=dims, m=dims, dtype=ti.f32)

#   @ti.func
#   def to_local(g: vec, b: basis):
#     return ti.math.inverse(b) @ g

#   @ti.func
#   def from_local(g: vec, b: basis):
#     return (b @ g)
  
#   return local_vector_adam_kernel(basis, to_local, from_local, betas, eps, dims)

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
    state['exp_avg'] = torch.zeros(*param.shape, dtype=torch.float32, device=param.device)
    state['exp_avg_sq'] = torch.zeros((param.shape[0],), dtype=torch.float32, device=param.device)

  return state


def get_scalar_state(state:dict, param: torch.Tensor):
  if len(state) == 0:
    state['exp_avg'] = torch.zeros_like(param)
    state['exp_avg_sq'] = torch.zeros_like(param)

  return state


def scalar_adam_step(group:dict, param: torch.Tensor, state: dict,

    visible_weight: torch.Tensor,
    visible_indexes: torch.Tensor,

  ):
  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 
  param_step = param.new_zeros(param.shape[0])

  kernel = adam_kernel(betas=group["betas"], eps=group["eps"])
  
  kernel(param_step, grad, 
         state['total_weight'], state['exp_avg'], state['exp_avg_sq'], 
         visible_indexes, visible_weight)
  
  return param_step


def vector_adam_step(group:dict, param: torch.Tensor, state: dict, 
                     visible_weight: torch.Tensor,
                     visible_indexes: torch.Tensor, 

                     basis: Optional[torch.Tensor]=None):
  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 
  param_step = param.new_zeros()

  dim = param.shape[1]

  kernel = vector_adam_kernel(betas=group["betas"], eps=group["eps"], dims=dim)

  if basis is not None:
    inv_basis = torch.linalg.inv(basis)
    grad = inv_basis @ grad

  kernel(param_step, grad, 
         state['total_weight'], state['exp_avg'], state['exp_avg_sq'], 
         visible_indexes, visible_weight)

  if basis is not None:
    param_step = basis @ param_step

  return param_step


@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))

class SparseAdam(torch.optim.Optimizer):
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
    running_vis[visible_indexes] = exp_lerp(self.vis_beta,running_vis[visible_indexes], visibility)

    weight = visibility / running_vis
    total_weight.index_add_(0, visible_indexes, weight)


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
    

    n = get_params(keys[0]).shape[0]
    self.update_visibility(visibility, visible_indexes, n)


    for k, group in named.items():
      param = get_params(k)
      if param.grad is None:
        continue
      
      assert param.shape[0] == n, f"param shape {param.shape[0]} != {n}"

      state = self.state[param]

      if group["type"] == "vector":
        vector_adam_step(group, param, get_vector_state(state, param), visible_indexes)
      elif group["type"] == "local_vector":
        assert basis is not None, "basis is required for basis optimizer"
        local_vector_adam_step(group, param, get_vector_state(state, param), visible_indexes, basis=basis)


      elif group["type"] == "scalar":
        scalar_adam_step(group, param, get_scalar_state(state, param), visible_indexes)
      else:
        raise ValueError(f"unknown group type {group['type']}")
    

