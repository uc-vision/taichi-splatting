from beartype import beartype
import taichi as ti
from taichi.math import mat4, mat3, vec3, vec4
import torch


import taichi_splatting.taichi_lib.f32 as lib
from .params import CameraParams
 

@ti.kernel
def frustum_culling_kernel(
    position: ti.types.ndarray(vec3, ndim=1),  # (N, 3)
    rotation: ti.types.ndarray(vec4, ndim=1),  # (N, 3)
    log_scale: ti.types.ndarray(vec3, ndim=1),  # (N, 3)

    T_camera_world: ti.types.ndarray(mat4, ndim=1),  # (1, 4, 4)

    projection: ti.types.ndarray(vec4, ndim=1),  # (1, 4)
    image_size: ti.math.ivec2,

    output_mask: ti.types.ndarray(ti.u1, ndim=1),  # (N), output
    
    near_plane: ti.f32,
    far_plane: ti.f32,

    gaussian_scale: ti.f32,
):    
    # filter points in camera
  for idx in range(position.shape[0]):
    uv, depth, uv_cov = lib.project_gaussian(
      T_camera_world[0], projection[0], image_size,
      position[idx], ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]))
    
    radius = lib.radii_from_cov(uv_cov) * gaussian_scale

    output_mask[idx] = (depth> near_plane and depth < far_plane and 
        uv.x >= -radius and uv.x < image_size.x + radius and 
        uv.y >= -radius and uv.y < image_size.y + radius)

@beartype
def frustum_culling(position: torch.Tensor, 
                    rotation: torch.Tensor,
                    log_scale: torch.Tensor,
                    
                    camera_params: CameraParams, 
                    gaussian_scale: float = 3.):
  mask = torch.empty(position.shape[0], dtype=torch.bool, device=position.device)
  

  frustum_culling_kernel(
    position=position.contiguous(),
    rotation=rotation.contiguous(),
    log_scale=log_scale.contiguous(),

    T_camera_world=camera_params.T_camera_world.unsqueeze(0),
    
    projection=camera_params.projection.unsqueeze(0),
    image_size=ti.math.ivec2(camera_params.image_size),

    output_mask=mask,

    near_plane=camera_params.near_plane,
    far_plane=camera_params.far_plane,
    
    gaussian_scale=gaussian_scale
  )

  return mask


