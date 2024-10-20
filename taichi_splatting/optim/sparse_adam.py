from functools import cache
import torch
import taichi as ti

@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)


@cache
def adam_kernel(betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, use_point_lr=False, use_mask_lr=False):
  b1, b2 = betas

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),  # N x D

             step: ti.types.ndarray(dtype=ti.f32, ndim=1), # N

             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             point_lr: ti.types.ndarray(dtype=ti.f32, ndim=1), # N learning rate multipliers across points
             mask_lr: ti.types.ndarray(dtype=ti.f32, ndim=1),  # D learning rate multipliers for each member of a param vector

             global_lr: ti.f32):

    for i in indexes:
      idx = indexes[i]
      lr_idx = point_lr[idx] if ti.static(use_point_lr) else 1.0

      step[idx] += 1 
      bias_factor = ti.sqrt(1 - b2 ** step[idx])  / (1 - b1 ** step[idx])


      for j in range(param.shape[1]):
        g = grad[idx, j]

        if ti.static(weight_decay != 0.0):
          g += weight_decay * param[idx, j]

        avg = lerp(b1, exp_avg[idx, j], g)
        avg_sq = lerp(b2, exp_avg_sq[idx, j], g * g)

        lr = global_lr * lr_idx * (mask_lr[j] if ti.static(use_mask_lr) else 1.0)
        param[idx, j] -= lr * avg / (ti.sqrt(avg_sq) + eps) * bias_factor

        exp_avg[idx, j] = avg
        exp_avg_sq[idx, j] = avg_sq


  return kernel

@cache
def vector_adam_kernel(betas=(0.9, 0.999), eps=1e-08, dims=3):
  b1, b2 = betas
  vec = ti.types.vector(n=dims, dtype=ti.f32)

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=vec, ndim=1), # N x D
             grad: ti.types.ndarray(dtype=vec, ndim=1),  # N x D

             step: ti.types.ndarray(dtype=ti.f32, ndim=1), # N

             exp_avg: ti.types.ndarray(dtype=vec, ndim=1),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             lr: ti.f32):

    for i in indexes:
      idx = indexes[i]

      step[idx] += 1
      bias_factor = ti.sqrt(1 - b2 ** step[idx])  / (1 - b1 ** step[idx])


      g = grad[idx]
      avg = lerp(b1, exp_avg[idx], g)

      norm = ti.math.dot(g, g)
      avg_sq = lerp(b2, exp_avg_sq[idx], norm)

      param[idx] -= lr * avg / (ti.sqrt(avg_sq) + eps) * bias_factor

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel


@cache
def local_vector_adam_kernel(basis_type, to_local, from_local, betas=(0.9, 0.999), eps=1e-08, dims=2):
  b1, b2 = betas
  vec = ti.types.vector(n=dims, dtype=ti.f32)
  

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=vec, ndim=1), # N 
             grad: ti.types.ndarray(dtype=vec, ndim=1),  # N 

             step: ti.types.ndarray(dtype=ti.f32, ndim=1), # N

             exp_avg: ti.types.ndarray(dtype=vec, ndim=1),    # N 
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1), # N 

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes
             basis: ti.types.ndarray(dtype=basis_type, ndim=1), # N 

             lr: ti.f32):

    for i in indexes:
      idx = indexes[i]

      step[idx] += 1
      bias_factor = ti.sqrt(1 - b2 ** step[idx])  / (1 - b1 ** step[idx])


      local_grad = to_local(grad[idx], basis[idx])
      avg = lerp(b1, exp_avg[idx], local_grad)

      norm = ti.math.dot(local_grad, local_grad)
      avg_sq = lerp(b2, exp_avg_sq[idx], norm)

      local_step = lr * avg / (ti.sqrt(avg_sq) + eps) * bias_factor
      param[idx] -= from_local(local_step, basis[idx])

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel

@cache
def basis_kernel(betas=(0.9, 0.999), eps=1e-08, dims=2):

  vec = ti.types.vector(n=dims, dtype=ti.f32)
  basis = ti.types.matrix(n=dims, m=dims, dtype=ti.f32)

  @ti.func
  def to_local(g: vec, b: basis):
    return ti.math.inverse(b) @ g

  @ti.func
  def from_local(g: vec, b: basis):
    return (b @ g)
  
  return local_vector_adam_kernel(basis, to_local, from_local, betas, eps, dims)


def scalar_adam_step(group:dict, param: torch.Tensor, state: dict, visible_indexes: torch.Tensor):
  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 


  if len(state) == 0:
    state['step'] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)
    state['exp_avg'] = torch.zeros_like(param)
    state['exp_avg_sq'] = torch.zeros_like(param)

  step = state["step"]
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
      
  kernel(param, grad, step, exp_avg, exp_avg_sq, visible_indexes, 
          point_lr=point_lr, mask_lr=mask_lr, global_lr=group["lr"])


def vector_adam_step(group:dict, param: torch.Tensor, state: dict, visible_indexes: torch.Tensor):
    

  grad = param.grad.view(param.shape[0], -1)
  param = param.view(param.shape[0], -1) 

  if len(state) == 0:
    state['step'] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)
    state['exp_avg'] = torch.zeros(*param.shape, dtype=torch.float32, device=param.device)
    state['exp_avg_sq'] = torch.zeros((param.shape[0],), dtype=torch.float32, device=param.device)

  step = state["step"]
  exp_avg = state["exp_avg"]
  exp_avg_sq = state["exp_avg_sq"]

  dim = param.shape[1]

  if group["basis"] is not None:
    basis = group["basis"]   
    if basis.shape != (param.shape[0], dim, dim):
      raise ValueError(f"basis shape {basis.shape} != {param.shape[0], dim, dim}")

    kernel = basis_kernel(betas=group["betas"], eps=group["eps"], dims=dim)
    kernel(param, grad, step, exp_avg, exp_avg_sq, visible_indexes, 
            basis=basis, lr=group["lr"])

  else:
    kernel = vector_adam_kernel(betas=group["betas"], eps=group["eps"], dims=dim)
    kernel(param, grad, step, exp_avg, exp_avg_sq, visible_indexes, lr=group["lr"])



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

    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, point_lr=None, mask_lr=None, basis=None, type="scalar")  
    super().__init__(params, defaults)

  def update_group(self, param, **kwargs):
    for group in self.param_groups:
      if param in group["params"]:
        group.update(kwargs)


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
      if group["type"] == "vector":
        vector_adam_step(group, param, state, visible_indexes)
      elif group["type"] == "scalar":
        scalar_adam_step(group, param, state, visible_indexes)
      else:
        raise ValueError(f"unknown group type {group['type']}")
    

