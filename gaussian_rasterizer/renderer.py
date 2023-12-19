
import taichi as ti
import torch

from gaussian_rasterizer.culling import CameraParams, frustum_culling
from gaussian_rasterizer.data_types import Gaussians
from gaussian_rasterizer.tile_mapper import map_to_tiles
from gaussian_rasterizer.projection import project_to_image
from gaussian_rasterizer.rasterizer import rasterize


def render_gaussians(
      gaussians: Gaussians,
      camera_params: CameraParams,
      tile_size: int = 16,
      margin_tiles: int = 3,
    ):
  
  gaussians = gaussians.contiguous()

  point_mask = frustum_culling(gaussians.position,
    camera_params=camera_params, margin_pixels=margin_tiles * tile_size
  )

  culled:Gaussians = gaussians[point_mask]
  gaussians2d, depths = project_to_image(culled, camera_params)

  overlap_to_point, ranges = map_to_tiles(gaussians2d, depths, 
    image_size=camera_params.image_size, tile_size=tile_size)
  
  n = culled.feature.shape[1]
  features_depth=torch.cat([culled.feature, depths.unsqueeze(1)], dim=1) 

  image = rasterize(gaussians=gaussians2d, features=features_depth, 
    tile_overlap_ranges=ranges, overlap_to_point=overlap_to_point,
    image_size=camera_params.image_size, tile_size=tile_size)
  
  return image[..., :n], image[..., n]