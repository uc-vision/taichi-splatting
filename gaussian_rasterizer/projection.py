
import taichi as ti
from taichi.math import  mat3, mat4
import torch
from gaussian_rasterizer.culling import CameraParams
from gaussian_rasterizer.data_types import Gaussians, Gaussian2D, Gaussian3D
from gaussian_rasterizer.ti import projection
from gaussian_rasterizer.ti.covariance import cov_to_conic



@ti.kernel
def project_to_image_kernel(  
  gaussians: ti.types.ndarray(Gaussian3D.vec, ndim=1),  # (N, 3 + feature_vec.n1) # input
  T_image_camera: ti.types.ndarray(mat3, ndim=1),  # (3, 3) camera projection

  T_camera_world: ti.types.ndarray(mat4, ndim=1),  # (1, 4, 4)
  
  points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (N, )
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

      # camera_origin = -t @ r.transpose()
      # ray_direction = positions[point_id] - camera_origin

      # get color by ray actually only cares about the direction of the ray, ray origin is not used
      # point_color = gaussian_point_3d.get_color_by_ray(
      #     ray_origin=ray_origin,
      #     ray_direction=ray_direction,
      # ) 
      
      points[idx] = Gaussian2D.to_vec(
          uv=uv,
          uv_conic=cov_to_conic(uv_cov),
          alpha=gaussian.alpha(),
      )

      depths[idx] = point_in_camera.z

def project_to_image(gaussians:Gaussians, camera_params: CameraParams):
  device = gaussians.position.device
  
  points = torch.empty((*gaussians.shape, Gaussian2D.vec.n), dtype=torch.float32, device=device)
  depths = torch.empty((*gaussians.shape, ), dtype=torch.float32, device=device)

  gaussians3d = torch.concat([gaussians.position, gaussians.log_scaling, gaussians.rotation, gaussians.alpha_logit], dim=-1)
  project_to_image_kernel(
      gaussians3d,

      T_image_camera=camera_params.T_image_camera,
      T_camera_world=camera_params.T_camera_world,
      points=points,
      depths=depths,
  )

  return points, depths

