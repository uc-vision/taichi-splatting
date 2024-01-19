
from numbers import Integral
from typing import Tuple
import torch
from taichi_splatting.data_types import Gaussians2D
from taichi_splatting.misc.encode_depth import encode_depth32

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
  
  image, alpha = rasterize(gaussians2d=gaussians2d, 
    encoded_depths= encode_depth32(gaussians.depth),
    features=gaussians.feature, 
    image_size=image_size, 
    config=raster_config)
  
  return image


