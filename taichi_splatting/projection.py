
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
def project_to_image_function(torch_dtype=torch.float32):
  dtype = torch_taichi[torch_dtype]

  lib = make_library(dtype)
  Gaussian3D, Gaussian2D = lib.Gaussian3D, lib.Gaussian2D



  @ti.kernel
  def project_to_image_kernel(  
    gaussians: ti.types.ndarray(Gaussian3D.vec, ndim=1),  # (N, 3 + feature_vec.n1) # input

    T_image_camera: ti.types.ndarray(ndim=2),  # (3, 3) camera projection
    T_camera_world: ti.types.ndarray(ndim=2),  # (4, 4)
    
    points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (N, 6)
    depths: ti.types.ndarray(dtype, ndim=1),  # (N, 2)
  ):

    for idx in range(gaussians.shape[0]):
        position, scale, rotation, alpha = Gaussian3D.unpack_activate(gaussians[idx])

        camera_image = lib.mat3_from_ndarray(T_image_camera)
        camera_world = lib.mat4_from_ndarray(T_camera_world)

        uv, point_in_camera = lib.point_to_camera(
            position, camera_world, camera_image)
      

        cov_in_camera = lib.gaussian_covariance_in_camera(
            camera_world, rotation, scale)

        uv_cov = lib.project_gaussian_to_image(
            camera_image, point_in_camera, cov_in_camera)

        depths[idx] = point_in_camera.z
        points[idx] = Gaussian2D.to_vec(
            uv=uv.xy,
            uv_conic=lib.cov_to_conic(uv_cov),
            alpha=alpha,
        )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians, T_image_camera, T_camera_world):
        dtype, device = T_image_camera.dtype, T_image_camera.device

        points = torch.empty((gaussians.shape[0], Gaussian2D.vec.n), dtype=dtype, device=device)
        depths = torch.empty((gaussians.shape[0], ), dtype=dtype, device=device)

        project_to_image_kernel(gaussians, 
              T_image_camera, T_camera_world,
              points, depths)
        
        ctx.save_for_backward(gaussians, T_image_camera, T_camera_world, points, depths)
        return points, depths

    @staticmethod
    def backward(ctx, dpoints, ddepths):
        gaussians,  T_image_camera, T_camera_world, points, depths = ctx.saved_tensors

        with restore_grad(gaussians,  T_image_camera, T_camera_world, points, depths):
          points.grad = dpoints.contiguous()
          depths.grad = ddepths.contiguous()
          project_to_image_kernel.grad(
            gaussians,  
            T_image_camera, T_camera_world, 
            points, depths)

          return gaussians.grad,  T_image_camera.grad, T_camera_world.grad

  return _module_function

@beartype
def apply(gaussians:torch.Tensor, T_image_camera:torch.Tensor, T_camera_world:torch.Tensor):
  _module_function = project_to_image_function(gaussians.dtype)
  return _module_function.apply(gaussians.contiguous(), 
        T_image_camera.contiguous(), T_camera_world.contiguous())

@beartype
def project_to_image(gaussians:torch.Tensor, camera_params: CameraParams) -> Tuple[torch.Tensor, torch.Tensor]:
  
  return apply(
      gaussians, camera_params.T_image_camera, camera_params.T_camera_world,
  )

