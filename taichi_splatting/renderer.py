

from dataclasses import dataclass
from beartype.typing import Optional
import torch

from taichi_splatting.data_types import check_packed3d
from taichi_splatting.misc.depth_variance import compute_depth_variance
from taichi_splatting.misc.encode_depth import encode_depth
from taichi_splatting.rasterizer import rasterize, RasterConfig
from taichi_splatting.spherical_harmonics import  evaluate_sh_at

from taichi_splatting.perspective import (
  frustum_culling, project_to_image, CameraParams)



@dataclass 
class Rendering:
  image: torch.Tensor  # (H, W, C)
  image_weight: torch.Tensor # (H, W, 1)
  
  depth: Optional[torch.Tensor] = None  # (H, W)
  depth_var: Optional[torch.Tensor] = None # (H, W)

  point_split_heuristics: Optional[torch.Tensor] = None  # (N, 1)

  



def render_gaussians(
  packed_gaussians: torch.Tensor,
  features: torch.Tensor,
  camera_params: CameraParams, 
  config:RasterConfig = RasterConfig(),      
  use_sh:bool = False,      
  render_depth:bool = False, 
  use_depth16:bool = False,
  compute_split_heuristics:bool = False
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
    use_depth16: bool - whether to use 16 bit depth encoding (otherwise 32 bit)
    compute_split_heuristics: bool - whether to compute the visibility for each point in the image
  
  Returns:
    images : Rendering - rendered images, with optional depth and depth variance and point weights
    
  """

  check_packed3d(packed_gaussians)      

  gaussians, features = cull_gaussians(packed_gaussians, features, camera_params, config.tile_size, config.margin_tiles)
  if use_sh:
    features = evaluate_sh_at(features, gaussians.detach(), camera_params.camera_position)
  else:
    assert len(features.shape) == 2, f"Features must be (N, C) if use_sh=False, got {features.shape}"

  gaussians2d, depthvars = project_to_image(gaussians, camera_params)
  depth_order = encode_depth(depthvars, 
    depth_range=(camera_params.near_plane, camera_params.far_plane),
    use_depth16 = use_depth16)
  
  if render_depth:
    features = torch.cat([depthvars, features], dim=1)
    
  raster = rasterize(gaussians2d, depth_order, features,
    image_size=camera_params.image_size, config=config, compute_split_heuristics=compute_split_heuristics)

  depth, depth_var = None, None
  feature_image = raster.image

  if render_depth:
    depth, depth_var = compute_depth_variance(raster.image, raster.image_weight)
    feature_image = feature_image[..., 3:]

  return Rendering(image=feature_image, 
                  image_weight=raster.image_weight, 
                  depth=depth, 
                  depth_var=depth_var, 
                  point_split_heuristics=raster.point_split_heuristics if compute_split_heuristics else None)



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

  indexes = torch.nonzero(point_mask, as_tuple=True)[0]
  return packed_gaussians[indexes], features[indexes]
