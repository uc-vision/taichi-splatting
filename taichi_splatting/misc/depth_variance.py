
from functools import cache
from beartype import beartype
import taichi as ti
import torch

from taichi_splatting.misc.autograd import restore_grad
from taichi_splatting.taichi_lib.conversions import torch_taichi


@cache
def depth_variance_func(torch_dtype=torch.float32, eps=1e-8):
  dtype = torch_taichi[torch_dtype]

  @ti.kernel
  def depth_var_kernel(  
    features_depth: ti.types.ndarray(dtype, ndim=3),  # (H, W, 3 + C) image features with 3 depth features at the start
    total_weight: ti.types.ndarray(dtype, ndim=2),  # (H, W) - pixel alpha (normalizing factor)
    depth: ti.types.ndarray(dtype, ndim=2),  # (H, W, ) # output
    depth_var: ti.types.ndarray(dtype, ndim=2),  # (H, W, ) # output
  ):
    h, w = features_depth.shape[0:2]

    for v, u in ti.ndrange(h, w):
      
      weight = total_weight[v, u] + eps
      d, d2, var = [features_depth[v, u, i] / weight
                     for i in ti.static(range(3))]
      
      depth[v, u] = d 
      depth_var[v, u] = (d2  - d**2) + var

  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features_depth, alpha):
      device = features_depth.device
      shape = features_depth.shape[:2]

      depth = torch.empty(shape, dtype=torch_dtype, device=device)
      depth_var = torch.empty(shape, dtype=torch_dtype, device=device)

      depth_var_kernel(features_depth, alpha, depth, depth_var)
      ctx.save_for_backward(features_depth, depth, depth_var)
      ctx.alpha = alpha

      return depth, depth_var

    @staticmethod
    def backward(ctx, ddepth, ddepth_var):
      features_depth, depth, depth_var = ctx.saved_tensors
      alpha = ctx.alpha

      with restore_grad(features_depth, depth, depth_var):
        depth.grad = ddepth.contiguous()
        depth_var.grad = ddepth_var.contiguous()
        depth_var_kernel.grad(
          features_depth, alpha, depth, depth_var)

        return features_depth.grad, None

  return _module_function



@beartype
def compute_depth_variance(features:torch.Tensor, alpha:torch.Tensor):
  """ 
  Compute depth and depth variance from image features.
  
  Parameters:
    features: torch.Tensor (N, 3 + C) - image features
    alpha:    torch.Tensor (N, 1) - alpha values
  """

  _module_function = depth_variance_func(features.dtype)
  return _module_function.apply(features.contiguous(), alpha.contiguous())


