
from functools import cache
from typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.autograd import restore_grad

from taichi_splatting.culling import CameraParams
from taichi_splatting.taichi_lib.generic import make_library
from taichi_splatting.taichi_lib.conversions import torch_taichi


@cache
def project_to_image_function(torch_dtype=torch.float32, perspective:bool = True):
  dtype = torch_taichi[torch_dtype]

  lib = make_library(dtype)
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

      uv_cov = lib.project_perspective_gaussian(
          camera_image, point_in_camera, cov_in_camera)

      depth_var[idx] = lib.vec3(point_in_camera.z, cov_in_camera[2, 2], point_in_camera.z ** 2)
      points[idx] = Gaussian2D.to_vec(
          uv=uv.xy,
          uv_conic=lib.cov_to_conic(uv_cov),
          alpha=alpha,
      )


  @ti.kernel
  def orthographic_projection_kernel(  
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

      uv, point_in_camera = lib.project_orthographic(
          position, camera_world, camera_image)
    
      cov_in_camera = lib.gaussian_covariance_in_camera(
          camera_world, rotation, scale)


      depth_var[idx] = ti.vec3(point_in_camera.z, cov_in_camera[2, 2], point_in_camera.z ** 2)
      points[idx] = Gaussian2D.to_vec(
          uv=uv.xy,
          uv_conic=lib.cov_to_conic(cov_in_camera[0:2, 0:2]),
          alpha=alpha,
      )

  projection_kernel = (project_perspective_kernel if perspective 
    else orthographic_projection_kernel)

  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians, T_image_camera, T_camera_world):
      dtype, device = T_image_camera.dtype, T_image_camera.device

      points = torch.empty((gaussians.shape[0], Gaussian2D.vec.n), dtype=dtype, device=device)
      depth_vars = torch.empty((gaussians.shape[0], 3), dtype=dtype, device=device)


      projection_kernel(gaussians, 
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
        projection_kernel.grad(
          gaussians,  
          T_image_camera, T_camera_world, 
          points, depth_vars)

        return gaussians.grad,  T_image_camera.grad, T_camera_world.grad

  return _module_function

@beartype
def apply(gaussians:torch.Tensor, T_image_camera:torch.Tensor, T_camera_world:torch.Tensor, orthographic:bool = False):
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
      orthographic=camera_params.orthographic
  )




@cache
def depth_var_func(torch_dtype=torch.float32, eps=1e-8):
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
def compute_depth_var(features:torch.Tensor, alpha:torch.Tensor):
  """ 
  Compute depth and depth variance from image features.
  
  Parameters:
    features: torch.Tensor (N, 3 + C) - image features
    alpha:    torch.Tensor (N, 1) - alpha values
  """

  _module_function = depth_var_func(features.dtype)
  return _module_function.apply(features.contiguous(), alpha.contiguous())



