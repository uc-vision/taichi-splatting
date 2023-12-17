
import taichi as ti

from gaussian_rasterizer.culling import CameraParams, frustum_culling
from gaussian_rasterizer.data_types import Gaussians
from gaussian_rasterizer import rasterizer
from gaussian_rasterizer.projection import project_to_image



def render_gaussians(
      gaussians: Gaussians,
      camera_params: CameraParams,
      tile_size: int = 16,
      margin_tiles: int = 3
    ):
  
  gaussians = gaussians.contiguous()

  point_mask = frustum_culling(
    pointcloud=gaussians.position,
    camera_params=camera_params,
    margin_pixels=margin_tiles * tile_size
  )
  culled:Gaussians = gaussians[point_mask]
  print(point_mask.shape, point_mask.sum())

  rasterize = rasterizer.rasterize(feature_type=ti.math.vec3, tile_size=tile_size)

  gaussians2d, depths = project_to_image(culled, camera_params)
  image = rasterize(gaussians2d, depths, culled.feature, camera_params.image_size)
