
from numbers import Integral
from typing import Tuple
import torch
from taichi_splatting.data_types import Gaussians2D

from taichi_splatting.tile_mapper import map_to_tiles, pad_to_tile
from taichi_splatting.rasterizer import rasterize, RasterConfig


def project_gaussians2d(points: Gaussians2D) -> torch.Tensor:
    scale = torch.exp(points.log_scaling)

    alpha = torch.sigmoid(points.alpha_logit)

    v1 = points.rotation / torch.norm(points.rotation, dim=1, keepdim=True)
    v2 = torch.stack([-v1[..., 1], v1[..., 0]], dim=-1)

    basis = torch.stack([v1, v2], dim=2) * scale.unsqueeze(-1)
    cov = torch.bmm(basis.transpose(1, 2), basis)
    
    inv_cov = torch.inverse(cov)

    conic = torch.stack([inv_cov[..., 0, 0], inv_cov[..., 0, 1], inv_cov[..., 1, 1]], dim=-1)
    return torch.cat([points.position, conic, alpha.unsqueeze(1)], dim=-1)  
    

def render_gaussians(
      gaussians: Gaussians2D,
      image_size: Tuple[Integral, Integral],
      raster_config: RasterConfig
    ):
  
  gaussians2d = project_gaussians2d(gaussians)
  padded_size = pad_to_tile(image_size, raster_config.tile_size)

  overlap_to_point, ranges = map_to_tiles(gaussians2d, gaussians.depth, 
    image_size=padded_size, config=raster_config)
  

  image, alpha = rasterize(gaussians=gaussians2d, features=gaussians.feature, 
    tile_overlap_ranges=ranges, overlap_to_point=overlap_to_point,
    image_size=image_size, config=raster_config)
  
  return image


