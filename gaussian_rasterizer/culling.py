from dataclasses import dataclass
import taichi as ti
from taichi.math import vec3, mat3, vec4, mat4
import torch

from gaussian_rasterizer.data_types import CameraParams

from .ti.transforms import qt_to_mat
from .ti.projection import point_to_camera



@ti.kernel
def frustum_culling_kernel(
    pointcloud: ti.types.ndarray(vec3, ndim=1),  # (N, 3)
    camera_intrinsics: ti.types.ndarray(mat3, ndim=1),  # (1, 3, 3)

    q_camera_pointcloud: ti.types.ndarray(vec4, ndim=1),  # (1, 4)
    t_camera_pointcloud: ti.types.ndarray(vec3, ndim=1),  # (1, 3)

    output_mask: ti.types.ndarray(ti.u1, ndim=1),  # (N), output
    
    near_plane: ti.f32,
    far_plane: ti.f32,

    image_size: ti.math.ivec2,
    margin_pixels: ti.i32,
):
    T_camera_pointcloud_mat = qt_to_mat(
        q=q_camera_pointcloud[0],
        t=t_camera_pointcloud[0],
    )

    
    # filter points in camera
    for point_id in range(pointcloud.shape[0]):
        pixel, point_in_camera = point_to_camera(
            position=pointcloud[point_id],
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics[0],
        )

        output_mask[point_id] = (point_in_camera.z > near_plane and 
            point_in_camera.z < far_plane and 
            pixel.x >= -margin_pixels and pixel.x < image_size.x + margin_pixels and 
                pixel.y >= -margin_pixels and pixel.y < image_size.y + margin_pixels)


def frustum_culling(pointcloud: torch.Tensor, camera_params: CameraParams, margin_pixels: int):
  mask = torch.zeros(pointcloud.shape[0], dtype=torch.bool, device=pointcloud.device)


  frustum_culling_kernel(
    pointcloud=pointcloud.contiguous(),
    camera_intrinsics=camera_params.camera_intrinsics.unsqueeze(0),
    q_camera_pointcloud=camera_params.q_camera_pointcloud,
    t_camera_pointcloud=camera_params.t_camera_pointcloud,
    output_mask=mask,

    near_plane=camera_params.near_plane,
    far_plane=camera_params.far_plane,
    image_size=ti.math.ivec2(camera_params.image_size),
    
    margin_pixels=margin_pixels
  )

  return mask
    
    