

from dataclasses import dataclass
from typing import Optional
import torch

from taichi_splatting.culling import CameraParams, frustum_culling
from taichi_splatting.data_types import check_packed3d
from taichi_splatting.projection import compute_depth_var, project_to_image
from taichi_splatting.rasterizer import rasterize, RasterConfig
from taichi_splatting.spherical_harmonics import  evaluate_sh_at


@dataclass 
class Rendering:
  image: torch.Tensor  # (H, W, C)
  depth: Optional[torch.Tensor] = None  # (H, W)
  depth_var: Optional[torch.Tensor] = None # (H, W)



def render_gaussians(
  packed_gaussians: torch.Tensor,
  features: torch.Tensor,
  camera_params: CameraParams, 
  config:RasterConfig,      
  use_sh:bool = False,      
  render_depth:bool = False 
) -> Rendering:
  """
  A complete renderer for 3D gaussians. 
  Parameters:
    packed_gaussians: torch.Tensor (N, 11) - packed 3D gaussians
    features: torch.Tensor (N, C) | torch.Tensor(N, 3, (D+1)**2) 
      features for each gaussian OR spherical harmonics coefficients of degree D
    
    camera_params: CameraParams
    config: RasterConfig
    use_sh: bool - whether to use spherical harmonics
    render_depth: bool - whether to render depth and depth variance
  
  Returns:
    images : Rendering - rendered images, with optional depth and depth variance
    
  """

  check_packed3d(packed_gaussians)      

  gaussians, features = cull_gaussians(packed_gaussians, features, camera_params, config.tile_size, config.margin_tiles)
  if use_sh:
    features = evaluate_sh_at(features, gaussians.detach(), camera_params.camera_position)
  else:
    assert len(features.shape) == 2, f"Features must be (N, C) if use_sh=False, got {features.shape}"

  gaussians2d, depthvars = project_to_image(gaussians, camera_params)

  if render_depth:
    features = torch.cat([depthvars, features], dim=1)
    
  image_features, total_weight = rasterize(gaussians2d, depthvars, features,
    image_size=camera_params.image_size, config=config)

  if render_depth:
    depth, depth_var = compute_depth_var(image_features, total_weight)
    return Rendering(image_features[:, :, 3:], depth, depth_var)

  return Rendering(image_features)



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