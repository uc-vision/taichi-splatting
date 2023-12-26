
from typing import Tuple
import taichi as ti
from taichi.math import  mat3, mat4
import torch
from taichi_splatting.autograd import restore_grad

from taichi_splatting.culling import CameraParams
from taichi_splatting.data_types import  Gaussian2D, Gaussian3D, Gaussians3D
from taichi_splatting.taichi_funcs import projection
from taichi_splatting.taichi_funcs.covariance import cov_to_conic



@ti.kernel
def project_to_image_kernel(  
  gaussians: ti.types.ndarray(Gaussian3D.vec, ndim=1),  # (N, 3 + feature_vec.n1) # input

  T_image_camera: ti.types.ndarray(mat3, ndim=1),  # (1, 3, 3) camera projection
  T_camera_world: ti.types.ndarray(mat4, ndim=1),  # (1, 4, 4)
  
  points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (N, 6)
  depths: ti.types.ndarray(ti.f32, ndim=1),  # (N, 2)
):

  for idx in range(gaussians.shape[0]):
      gaussian = Gaussian3D.from_vec(gaussians[idx])
      uv, point_in_camera = projection.point_to_camera(
          position=gaussian.position,
          T_camera_world=T_camera_world[0],
          projective_transform=T_image_camera[0],
      )

      cov_in_camera = projection.gaussian_covariance_in_camera(
          T_camera_world[0], ti.math.normalize(gaussian.rotation), gaussian.scale())

      uv_cov = projection.project_gaussian_to_image(
          T_image_camera[0], point_in_camera, cov_in_camera)

      points[idx] = Gaussian2D.to_vec(
          uv=uv,
          uv_conic=cov_to_conic(uv_cov),
          alpha=gaussian.alpha(),
      )

      depths[idx] = point_in_camera.z




class _module_function(torch.autograd.Function):
  @staticmethod
  def forward(ctx, gaussians, T_image_camera, T_camera_world):
      points = torch.empty((gaussians.shape[0], Gaussian2D.vec.n), dtype=torch.float32, device=gaussians.device)
      depths = torch.empty((gaussians.shape[0], ), dtype=torch.float32, device=gaussians.device)

      project_to_image_kernel(gaussians, 
            T_image_camera.unsqueeze(0), T_camera_world.unsqueeze(0),
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
          T_image_camera.unsqueeze(0), T_camera_world.unsqueeze(0), 
          points, depths)

        return gaussians.grad,  T_image_camera.grad, T_camera_world.grad

def apply(gaussians, T_image_camera, T_camera_world):
  return _module_function.apply(gaussians.contiguous(), 
        T_image_camera.contiguous(), T_camera_world.contiguous())

def project_to_image(gaussians:Gaussians3D, camera_params: CameraParams) -> Tuple[torch.Tensor, torch.Tensor]:
  
  gaussians3d = torch.concat([gaussians.position, 
        gaussians.log_scaling, gaussians.rotation, gaussians.alpha_logit], dim=-1)
  return apply(
      gaussians3d, camera_params.T_image_camera, camera_params.T_camera_world,
  )

