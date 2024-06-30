from functools import cache
import torch
import taichi as ti


@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)



@cache
def adam_vector_kernel(n:int, betas=(0.9, 0.999), eps=1e-08):
  b1, b2 = betas
  dtype = ti.types.vector(n, ti.f32)

  @ti.kernel
  def k(
      param: ti.types.ndarray(dtype=dtype, ndim=1),
      grad: ti.types.ndarray(dtype=dtype, ndim=1),

      exp_avg: ti.types.ndarray(dtype=dtype, ndim=1),
      exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1),

      indexes: ti.types.ndarray(dtype=ti.int64, ndim=1),
      lr: ti.f32):

    for i in indexes:
      idx = indexes[i]
      grad_norm = (grad[idx] * grad[idx]).sum()

      avg = lerp(b1, exp_avg[idx], grad[idx])
      avg_sq = lerp(b2, exp_avg_sq[idx], grad_norm)

      step = lr * avg / (ti.sqrt(avg_sq) + eps)

      param[idx] -= step
      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return k


class SparseVectorAdam(torch.optim.Optimizer):
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
    if not 0.0 <= lr:
      raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

    defaults = dict(lr=lr, betas=betas, eps=eps)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, visibility: torch.Tensor):
    for group in self.param_groups:

      assert len(group["params"]) == 1, "more than one tensor in group"
      param = group["params"][0]
      if param.grad is None:
        continue

      if param.dim() == 1:
        param, grad = param.unsqueeze(1), param.grad.unsqueeze(1)
      else:
        grad = param.grad

      
      # Lazy state initialization
      state = self.state[param]
      if len(state) == 0:
        state['step'] = torch.tensor(0.0, dtype=torch.float32)
        state['exp_avg'] = torch.zeros_like(param)
        state['exp_avg_sq'] = torch.zeros(param.shape[0], device=param.device, dtype=param.dtype)

      exp_avg = state["exp_avg"]
      exp_avg_sq = state["exp_avg_sq"]


      kernel = adam_vector_kernel(n=param.shape[1],
                      betas=group["betas"], eps=group["eps"])
      
      kernel(param, grad, exp_avg, exp_avg_sq, visibility, group["lr"])
