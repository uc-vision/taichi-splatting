
from functools import cache
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.misc.autograd import restore_grad

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
def project_to_image_function(torch_dtype=torch.float32, 
                              blur_cov:float = 0.3):
  dtype = torch_taichi[torch_dtype]

  lib = get_library(dtype)
  Gaussian3D, Gaussian2D = lib.Gaussian3D, lib.Gaussian2D



  @ti.kernel
  def project_perspective_kernel(  
    gaussians: ti.types.ndarray(Gaussian3D.vec, ndim=1),  # (N, 3 + feature_vec.n1) # input

    T_image_camera: ti.types.ndarray(ndim=2),  # (3, 3) camera projection
    T_camera_world: ti.types.ndarray(ndim=2),  # (4, 4)
    
    points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (N, 6)
    depth_var: ti.types.ndarray(lib.vec3, ndim=1),  # (N, 3)
  ):

    for idx in range(gaussians.shape[0]):
      position, scale, rotation, alpha = Gaussian3D.unpack_activate(gaussians[idx])

      camera_image = lib.mat3_from_ndarray(T_image_camera)
      camera_world = lib.mat4_from_ndarray(T_camera_world)

      uv, point_in_camera = lib.project_perspective_camera_image(
          position, camera_world, camera_image)
    
      cov_in_camera = lib.gaussian_covariance_in_camera(
          camera_world, rotation, scale)

      uv_cov = lib.upper(lib.project_perspective_gaussian(
          camera_image, point_in_camera, cov_in_camera))
      
      # add small fudge factor blur to avoid numerical issues
      uv_cov += lib.vec3([blur_cov, 0, blur_cov]) 
      uv_conic = lib.inverse_cov(uv_cov)

      depth_var[idx] = lib.vec3(point_in_camera.z, cov_in_camera[2, 2], point_in_camera.z ** 2)

      points[idx] = Gaussian2D.to_vec(
          uv=uv.xy,
          uv_conic=uv_conic,
          alpha=alpha,
      )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians, T_image_camera, T_camera_world):
      dtype, device = T_image_camera.dtype, T_image_camera.device

      points = torch.empty((gaussians.shape[0], Gaussian2D.vec.n), dtype=dtype, device=device)
      depth_vars = torch.empty((gaussians.shape[0], 3), dtype=dtype, device=device)


      project_perspective_kernel(gaussians, 
            T_image_camera, T_camera_world,
            points, depth_vars)
      
      ctx.save_for_backward(gaussians, T_image_camera, T_camera_world, points, depth_vars)
      return points, depth_vars

    @staticmethod
    def backward(ctx, dpoints, ddepth_vars):
      gaussians,  T_image_camera, T_camera_world, points, depth_vars = ctx.saved_tensors

      with restore_grad(gaussians,  T_image_camera, T_camera_world, points, depth_vars):
        points.grad = dpoints.contiguous()
        depth_vars.grad = ddepth_vars.contiguous()
        project_perspective_kernel.grad(
          gaussians,  
          T_image_camera, T_camera_world, 
          points, depth_vars)

        return gaussians.grad,  T_image_camera.grad, T_camera_world.grad

  return _module_function

@beartype
def apply(gaussians:torch.Tensor, T_image_camera:torch.Tensor, T_camera_world:torch.Tensor):
  _module_function = project_to_image_function(gaussians.dtype)
  return _module_function.apply(gaussians.contiguous(), 
        T_image_camera.contiguous(), T_camera_world.contiguous())

@beartype
def project_to_image(gaussians:torch.Tensor, camera_params: CameraParams
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ 
  Project 3D gaussians to 2D gaussians in image space using perspective projection.
  Use EWA approximation for the projection of the gaussian covariance,
  as described in Zwicker, et al. "EWA splatting." 2003.
  
  Parameters:
    gaussians: torch.Tensor (N, 11) - packed 3D gaussians
    camera_params: CameraParams

  Returns:
    points:    torch.Tensor (N, 6)  - packed 2D gaussians in image space
    depth_var: torch.Tensor (N, 3)  - depth, depth variance and depth^2 of gaussians
  """

  return apply(
      gaussians, 
      camera_params.T_image_camera, 
      camera_params.T_camera_world,
  )




