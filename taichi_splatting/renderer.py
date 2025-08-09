

from dataclasses import replace
from beartype import beartype
import torch

from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.mapper.tile_mapper import map_to_tiles
from taichi_splatting.rasterizer import RasterConfig
from taichi_splatting.rasterizer.function import rasterize_with_tiles
from taichi_splatting.rendering import RenderedPoints, Rendering

from taichi_splatting.perspective import (CameraParams)

from taichi_splatting.perspective.projection import project_to_image
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.torch_lib.projection import ndc_depth


@beartype
def render_gaussians(
  gaussians: Gaussians3D,
  camera_params: CameraParams, 
  config:RasterConfig = RasterConfig(),      
  use_sh:bool = False,      
  render_depth:bool = False, 
  use_depth16:bool = False,
  render_median_depth:bool = False
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
  
  Returns:
    images : Rendering - rendered images, with optional depth and depth variance and point weights
    
  """

  gaussians2d, depths, indexes = project_to_image(gaussians, camera_params, config)

  if use_sh:
    features = evaluate_sh_at(gaussians.feature, gaussians.position.detach(), indexes, camera_params.camera_position)
  else:
    features = gaussians.feature[indexes]
    assert len(features.shape) == 2, f"Features must be (N, C) if use_sh=False, got {features.shape}"

  return render_projected(indexes, gaussians2d, features, depths, camera_params, config, 
                  use_depth16=use_depth16, render_median_depth=render_median_depth)


def render_projected(indexes:torch.Tensor, gaussians2d:torch.Tensor, 
      features:torch.Tensor, depths:torch.Tensor, 
      camera_params: CameraParams, config:RasterConfig,      
      use_depth16:bool = False, render_median_depth:bool = False) -> Rendering:

  ndc_depths = TaichiQueue.run_sync(ndc_depth, depths, camera_params.near_plane, camera_params.far_plane)
  
  overlap_to_point, tile_overlap_ranges = map_to_tiles(gaussians2d, ndc_depths, 
    image_size=camera_params.image_size, config=config, use_depth16=use_depth16)
  
  raster = rasterize_with_tiles(gaussians2d, features, 
    tile_overlap_ranges=tile_overlap_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
    image_size=camera_params.image_size, config=config)

  median_depth = None
  if render_median_depth:
    raster_depth = rasterize_with_tiles(gaussians2d, depths, 
      tile_overlap_ranges=tile_overlap_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
      image_size=camera_params.image_size, config=replace(config, use_alpha_blending=False, saturate_threshold=0.5))
    
    median_depth = raster_depth.image.squeeze(-1)

  img_depth = None
  feature_image = raster.image

  point_shape = (depths.shape[0],)


  points = RenderedPoints(idx=indexes,
                  depths=depths,
                  gaussians2d=gaussians2d,

                  _visibility = raster.visibility if config.compute_visibility else None,  
                  _prune_cost=raster.point_heuristic[:, 0] if config.compute_point_heuristic else None,
                  _split_score=raster.point_heuristic[:, 1] if config.compute_point_heuristic else None,
                  features=features, 
                  attributes=None,
                  batch_size=point_shape)

  return Rendering(image=feature_image, 
                  image_weight=raster.image_weight,  
                  depth_image=img_depth, 
                  median_depth_image=median_depth,

                  points=points,
                  camera=camera_params,
                  config=config)

  


def viewspace_gradient(gaussians2d: torch.Tensor):
  assert gaussians2d.shape[1] == 6, f"Expected 2D gaussians, got {gaussians2d.shape}"
  assert gaussians2d.grad is not None, "Expected gradients on gaussians2d, run backward first with gaussians2d.retain_grad()"

  xy_grad = gaussians2d.grad[:, :2]
  return torch.norm(xy_grad, dim=1)



