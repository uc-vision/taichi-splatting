from functools import cache
import torch
import taichi as ti

from taichi_splatting.taichi_lib.f32 import quat_to_mat


@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)


@cache
def adam_kernel(betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, use_point_lr=False, use_mask_lr=False):
  b1, b2 = betas

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),  # N x D

             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             point_lr: ti.types.ndarray(dtype=ti.f32, ndim=1), # N learning rate multipliers across points
             mask_lr: ti.types.ndarray(dtype=ti.f32, ndim=1),  # D learning rate multipliers for each member of a param vector

             global_lr: ti.f32):

    for i in indexes:
      idx = indexes[i]
      lr_idx = point_lr[idx] if ti.static(use_point_lr) else 1.0

      for j in range(param.shape[1]):
        g = grad[idx, j]

        if ti.static(weight_decay != 0.0):
          g += weight_decay * param[idx, j]

        avg = lerp(b1, exp_avg[idx, j], g)
        avg_sq = lerp(b2, exp_avg_sq[idx, j], g * g)

        lr = global_lr * lr_idx * (mask_lr[j] if ti.static(use_mask_lr) else 1.0)
        param[idx, j] -= lr * avg / (ti.sqrt(avg_sq) + eps)

        exp_avg[idx, j] = avg
        exp_avg_sq[idx, j] = avg_sq

  return kernel

@cache
def position_3d_kernel(betas=(0.9, 0.999), eps=1e-08):
  b1, b2 = betas

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=ti.math.vec3, ndim=1), # N x 3
             grad: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),  # N x 3

             exp_avg: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),    # N x 3
             exp_avg_sq: ti.types.ndarray(dtype=ti.math.vec3, ndim=1), # N x 3

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             log_scale: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
             rotation: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),

             lr: ti.f32):

    for i in indexes:
      idx = indexes[i]

      m = quat_to_mat(ti.math.normalize(rotation[idx]))
      scale = ti.math.exp(log_scale[idx])

      local_grad = (m.transpose() @ grad[idx]) / scale

      avg = lerp(b1, exp_avg[idx], local_grad)
      avg_sq = lerp(b2, exp_avg_sq[idx], ti.math.length_sq(local_grad))

      local_step = lr * avg / (ti.sqrt(avg_sq) + eps)
      param[idx] -= m @ (local_step * scale)

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel


@cache
def position_2d_kernel(betas=(0.9, 0.999), eps=1e-08):
  b1, b2 = betas

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=ti.math.vec2, ndim=1), # N x 2
             grad: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),  # N x 2

             exp_avg: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),    # N x 2
             exp_avg_sq: ti.types.ndarray(dtype=ti.math.vec2, ndim=1), # N x 2

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             log_scale: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),
             rotation: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),

             lr: ti.f32):

    for i in indexes:
      idx = indexes[i]

      m = quat_to_mat(ti.math.normalize(rotation[idx]))
      scale = ti.math.exp(log_scale[idx])

      local_grad = (m.transpose() @ grad[idx]) / scale

      avg = lerp(b1, exp_avg[idx], local_grad)
      avg_sq = lerp(b2, exp_avg_sq[idx], ti.math.length_sq(local_grad))

      local_step = lr * avg / (ti.sqrt(avg_sq) + eps)
      param[idx] -= m @ (local_step * scale)

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel


class SparseAdam(torch.optim.Optimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0):
    
    if not 0.0 <= lr:
      raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon value:point_lr {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, point_lr=None, mask_lr=None, type="adam")  
    super().__init__(params, defaults)

  def update_group(self, param, **kwargs):
    for group in self.param_groups:
      if param in group["params"]:
        group.update(kwargs)


  def adam_step(self, group:dict, param: torch.Tensor, state: dict, visible_indexes: torch.Tensor):
    grad = param.grad.view(param.shape[0], -1)
    param = param.view(param.shape[0], -1) 


    if len(state) == 0:
      state['step'] = torch.tensor(0.0, dtype=torch.float32)
      state['exp_avg'] = torch.zeros_like(param)
      state['exp_avg_sq'] = torch.zeros_like(param)

    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    kernel = adam_kernel(betas=group["betas"], eps=group["eps"], 
                        weight_decay=group["weight_decay"], 
                        use_point_lr=group["point_lr"] is not None, 
                        use_mask_lr=group["mask_lr"] is not None)
    
    if group["point_lr"] is not None:
      point_lr=group["point_lr"]
      assert point_lr.shape == (param.shape[0],), f"point_lr shape {point_lr.shape} != {param.shape[0]}"
    else:
      point_lr = torch.empty((0,), device=param.device, dtype=torch.float32)

    if group["mask_lr"] is not None:

      mask_lr = group["mask_lr"].view(-1)
      assert mask_lr.shape == param.shape[1:], f"mask_lr shape {mask_lr.shape} != {param.shape[1:]}"

    else:
      mask_lr = torch.empty((0,), device=param.device, dtype=torch.float32)
        
    kernel(param, grad, exp_avg, exp_avg_sq, visible_indexes, 
            point_lr=point_lr, mask_lr=mask_lr, global_lr=group["lr"])




  def position_step(self, group:dict, param: torch.Tensor, state: dict, visible_indexes: torch.Tensor, 
                    log_scale: torch.Tensor, rotation: torch.Tensor):
    
    grad = param.grad.view(param.shape[0], 3)
    param = param.view(param.shape[0], 3) 

    if len(state) == 0:
      state['step'] = torch.tensor(0.0, dtype=torch.float32)
      state['exp_avg'] = torch.zeros(*param.shape, dtype=torch.float32)
      state['exp_avg_sq'] = torch.zeros(param.shape[0], 1, dtype=torch.float32)

    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    if param.shape[1] == 3:
      kernel = position_3d_kernel(betas=group["betas"], eps=group["eps"])
    else:
      kernel = position_2d_kernel(betas=group["betas"], eps=group["eps"])
            
    kernel(param, grad, exp_avg, exp_avg_sq, visible_indexes, 
            log_scale=log_scale, rotation=rotation, lr=group["lr"])




  @torch.no_grad()
  def step(self, visible_indexes: torch.Tensor):
    named = {group["name"]: group for group in self.param_groups}
    
    def get_params(k):
      group = named[k]

      n = len(group["params"])
      assert n == 1, f"expected 1 tensor in group {k}, got {n}"
      return group["params"][0]

    for k, group in named.items():
      param = get_params(k)
      if param.grad is None:
        continue
      
      state = self.state[param]
      if group["type"] == "position":
        self.position_step(group, param, state, visible_indexes,
                           log_scale=get_params("log_scale"),
                           rotation=get_params("rotation"))
      else:
        self.adam_step(group, param, state, visible_indexes)
    

