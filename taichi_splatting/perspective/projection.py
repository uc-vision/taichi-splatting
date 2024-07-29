
from functools import cache
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.optim.autograd import restore_grad

from .params import CameraParams
from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.conversions import torch_taichi

# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()

import warnings
warnings.filterwarnings('ignore', '(.*)that is not a leaf Tensor is being accessed(.*)') 


@cache
def project_to_image_function(torch_dtype=torch.float32):
  dtype = torch_taichi[torch_dtype]
  lib = get_library(dtype)


  @ti.kernel
  def project_perspective_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)

    indexes: ti.types.ndarray(ti.i64, ndim=1),  # (N) indexes of points to render from 0 to M

    T_camera_world: ti.types.ndarray(ndim=2),  # (4, 4)

    projection: ti.types.ndarray(ndim=1),  # (4) camera projection (fx, fy, cx, cy)
    image_size: lib.vec2,  # (2) image size
    
    points: ti.types.ndarray(lib.Gaussian2D.vec, ndim=1),  # (N, 6)
    depth: ti.types.ndarray(lib.vec1, ndim=1),  # (N, 1)
    blur_cov:ti.f32
  ):

    for i in range(indexes.shape[0]):
      idx = indexes[i]

      uv, z, uv_cov = lib.project_gaussian(
        lib.mat4_from_ndarray(T_camera_world), lib.vec4_from_ndarray(projection), image_size,
        position[idx], ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]))

      # add small fudge factor blur to avoid numerical issues
      uv_cov += lib.vec3([blur_cov, 0, blur_cov]) 
      uv_conic = lib.inverse_cov(uv_cov)

      depth[i] = z
      points[i] = lib.Gaussian2D.to_vec(
          uv=uv,
          uv_conic=uv_conic,
          alpha=lib.sigmoid(alpha_logit[idx][0]),
      )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, log_scaling, rotation, alpha_logit,
                indexes,
                T_camera_world,
                projection, image_size,
                blur_cov):
      dtype, device = projection.dtype, projection.device

      n = indexes.shape[0]

      points = torch.empty((n, lib.Gaussian2D.vec.n), dtype=dtype, device=device)
      depth = torch.empty((n, 1), dtype=dtype, device=device)

      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)


      project_perspective_kernel(*gaussian_tensors, 
            indexes,
            T_camera_world,
            projection, lib.vec2(image_size),
            points, depth,
            blur_cov)
      
      ctx.indexes = indexes
      ctx.blur_cov = blur_cov
      ctx.image_size = image_size
      
      ctx.mark_non_differentiable(indexes)
      ctx.save_for_backward(*gaussian_tensors,
         T_camera_world, 
         projection, 
         points, depth)
      
      return points, depth

    @staticmethod
    def backward(ctx, dpoints, ddepth):

      gaussian_tensors = ctx.saved_tensors[:4]
      T_camera_world, projection, points, depth = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors,  projection, T_camera_world, points, depth):
        points.grad = dpoints.contiguous()
        depth.grad = ddepth.contiguous()
        
        project_perspective_kernel.grad(
          *gaussian_tensors,  
          ctx.indexes,
          T_camera_world, 
          projection, lib.vec2(ctx.image_size),
          points, depth,
          ctx.blur_cov)

        return (*[tensor.grad for tensor in gaussian_tensors], 
                None, T_camera_world.grad, projection.grad,
                None, None)

  return _module_function

@beartype
def apply(position:torch.Tensor, log_scaling:torch.Tensor,
          rotation:torch.Tensor, alpha_logit:torch.Tensor,
          indexes:torch.Tensor,
          T_camera_world:torch.Tensor,
          projection:torch.Tensor, 
          
          image_size:Tuple[Integral, Integral],
          blur_cov:float=0.3):
  
  _module_function = project_to_image_function(position.dtype)
  return _module_function.apply(
    position.contiguous(),
    log_scaling.contiguous(),
    rotation.contiguous(),
    alpha_logit.contiguous(),
    indexes.contiguous(),

    T_camera_world.contiguous(),
    projection.contiguous(),
    image_size,
    
    blur_cov)

@beartype
def project_to_image(gaussians:Gaussians3D, indexes:torch.Tensor, camera_params: CameraParams, 
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ 
  Project 3D gaussians to 2D gaussians in image space using perspective projection.
  Use EWA approximation for the projection of the gaussian covariance,
  as described in Zwicker, et al. "EWA splatting." 2003.
  
  Parameters:
    gaussians3D: 3D gaussian representation tensorclass
    camera_params: CameraParams

  Returns:
    points:    torch.Tensor (N, 6)  - packed 2D gaussians in image space
    depths:    torch.Tensor (N, 1)  - depth, depth variance and depth^2 of gaussians
  """

  return apply(
      *gaussians.shape_tensors(),
      indexes,
      camera_params.T_camera_world,

      camera_params.projection,
      camera_params.image_size,

      blur_cov = camera_params.blur_cov
  )




