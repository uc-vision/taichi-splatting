
from numbers import Integral
from typing import Tuple
import torch

from taichi_splatting.culling import CameraParams, frustum_culling
from taichi_splatting.data_types import check_packed3d
from taichi_splatting.tile_mapper import  map_to_tiles, pad_to_tile
from taichi_splatting.projection import project_to_image
from taichi_splatting.rasterizer import rasterize, RasterConfig
from taichi_splatting.spherical_harmonics import  evaluate_sh_at




def render_sh_gaussians(
  packed_gaussians: torch.Tensor,
  sh_features: torch.Tensor,
  camera_params: CameraParams,
  config:RasterConfig
):
  check_packed3d(packed_gaussians)      

      
  gaussians, sh_features = cull_gaussians(packed_gaussians, sh_features, camera_params, config.tile_size, config.margin_tiles)

  features = evaluate_sh_at(sh_features, gaussians.detach(), camera_params.camera_position)
  gaussians2d, depths = project_to_image(gaussians, camera_params)

  return _render_with_features(gaussians2d, depths, features,
    image_size=camera_params.image_size, config=config)


def render_gaussians(
      packed_gaussians: torch.Tensor,
      features : torch.Tensor,
      camera_params: CameraParams,
      config:RasterConfig
    ):
  check_packed3d(packed_gaussians)      

  
  gaussians, features = cull_gaussians(packed_gaussians, features, camera_params, config.tile_size, config.margin_tiles)
  gaussians2d, depths = project_to_image(gaussians, camera_params)

  return _render_with_features(gaussians2d, depths, features,
    image_size=camera_params.image_size, config=config)

def _render_with_features(gaussians2d:torch.Tensor, depths:torch.Tensor, 
                          features:torch.Tensor, image_size:Tuple[Integral, Integral],
                          config:RasterConfig):
    
  # render with padding to tile_size, later crop back to original size
  padded_size = pad_to_tile(image_size, config.tile_size)
  overlap_to_point, ranges = map_to_tiles(gaussians2d, depths, 
    image_size=padded_size, config=config)

  n = features.shape[1]
  features=torch.cat([features, depths.unsqueeze(1)], dim=1) 

  image, alpha = rasterize(gaussians=gaussians2d, features=features, 
    tile_overlap_ranges=ranges, overlap_to_point=overlap_to_point,
    image_size=image_size, config=config)

  depth = image[..., n] / (alpha + 1e-6)
  return image[..., :n], depth


def cull_gaussians(
  packed_gaussians: torch.Tensor,  # packed gaussians
  features: torch.Tensor,  
  camera_params: CameraParams,
  tile_size: int = 16,
  margin_tiles: int = 3
):
  point_mask = frustum_culling(packed_gaussians,
    camera_params=camera_params, margin_pixels=margin_tiles * tile_size
  )

  return packed_gaussians[point_mask], features[point_mask]