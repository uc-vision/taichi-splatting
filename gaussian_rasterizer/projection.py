
import taichi as ti
from taichi.math import vec3, mat3, vec4
import torch
from gaussian_rasterizer.culling import CameraParams
from gaussian_rasterizer.data_types import Gaussians, Gaussian2D, Gaussian3D
from gaussian_rasterizer.taichi import transforms, projection
from gaussian_rasterizer.taichi.covariance import cov_to_conic



@ti.kernel
def project_to_image_kernel(  
  gaussians: ti.types.ndarray(Gaussian3D.vec, ndim=1),  # (N, 3 + feature_vec.n1) # input
  camera_intrinsics: ti.types.ndarray(mat3, ndim=1),  # (3, 3)

  q_camera_pointcloud: ti.types.ndarray(vec4, ndim=1),  # (1, 4)
  t_camera_pointcloud: ti.types.ndarray(vec3, ndim=1),  # (1, 3)
  
  points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (N, )
  depths: ti.types.ndarray(ti.f32, ndim=1),  # (N, 2)
):

  for idx in range(gaussians.shape[0]):
      gaussian = Gaussian3D.from_vec(gaussians[idx])
      
      r = transforms.quat_to_mat(q_camera_pointcloud[0])
      t = t_camera_pointcloud[0]

      T_camera_world=transforms.join_rt(r, t)
      # camera_origin = -t @ r.transpose()
      intrinsics_mat = camera_intrinsics[0]

      uv, point_in_camera = projection.point_to_camera(
          position=gaussian.position,
          T_camera_world=T_camera_world,
          projective_transform=intrinsics_mat,
      )

      cov_in_camera = projection.gaussian_covariance_in_camera(
          T_camera_world, ti.math.normalize(gaussian.rotation), gaussian.scale())

      uv_cov = projection.project_gaussian_to_image(
          intrinsics_mat, point_in_camera, cov_in_camera)

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

      camera_intrinsics=camera_params.camera_intrinsics.unsqueeze(0),
      q_camera_pointcloud=camera_params.q_camera_pointcloud,
      t_camera_pointcloud=camera_params.t_camera_pointcloud,
      points=points,
      depths=depths,
  )

  return points, depths

