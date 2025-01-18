from dataclasses import dataclass, fields
from functools import cached_property
from numbers import Integral
from typing import Any, Optional, Tuple
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.perspective.params import CameraParams
import torch

from taichi_splatting.torch_lib.projection import ndc_depth
from tensordict import tensorclass


def unpack(dc) -> dict[str, Any]:
    return {field.name:getattr(dc, field.name) for field in fields(dc)}

@tensorclass
class RenderedPoints:

  idx: torch.Tensor # 1, index of points in larger scene
  depths: torch.Tensor # 1, point depths

  gaussians2d: torch.Tensor # 7, gaussians2d, 2d gaussians after projection

  _prune_cost: Optional[torch.Tensor] = None # 1, prune cost - heuristic for point pruning computed in backward pass
  _split_score: Optional[torch.Tensor] = None # 1, split score - heuristic for point splitting computed in backward pass

  _visibility: Optional[torch.Tensor] = None # 1, visibility - sum of pre-multiplied alpha values during rasterization

  @property 
  def prune_cost(self):
    assert self._prune_cost is not None, "No prune cost information available (render with config.compute_point_heuristic=True)"
    return self._prune_cost

  @property
  def split_score(self):
    assert self._split_score is not None, "No split score information available (render with config.compute_point_heuristic=True)"
    return self._split_score
  
  @property
  def visibility(self):
    assert self._visibility is not None, "No visibility information available (render with config.compute_visibility=True)"
    return self._visibility

  @property
  def screen_scale(self):
    return self.gaussians2d[:, 4:6]

  @property
  def opacity(self):
    return self.gaussians2d[:, 6]
  
  @cached_property
  def visible(self) -> 'RenderedPoints':
    return self[self.visible_mask]
  
  @property
  def visible_mask(self) -> torch.Tensor:
    return self._visibility > 0.0
  
  def gaussian_scale(self, alpha_threshold:float=1.0/255):
    """ Factor of the gaussian bounds used for culling,
     Original gaussian splatting uses fixed gaussian_scale = 3.0
   """
    return torch.sqrt(2 * torch.log(self.opacity / alpha_threshold))
  
  @property
  def ndc(self, near:float, far:float):
    return ndc_depth(self.depths, near, far)
  
  def detach(self):
    return self.apply(torch.Tensor.detach)



@dataclass(frozen=True, kw_only=True)
class Rendering:
  """ Collection of outputs from the renderer, 

  depth and depth var are optional, as they are only computed if render_depth=True
  point_heuristic is computed in the backward pass if compute_point_heuristic=True

  """
  image: torch.Tensor        # (H, W, C) - rendered image, C channels of features
  image_weight: torch.Tensor # (H, W, 1) - weight of each pixel (total alpha)

  depth_image: Optional[torch.Tensor] = None      # (H, W)    - depth map 
  median_depth_image: Optional[torch.Tensor] = None  # (H, W) - median depth map

  points: RenderedPoints # (N,) renderered points which were in view

  camera : CameraParams
  config: RasterConfig

  @cached_property
  def ndc_image(self) -> torch.Tensor:
    return ndc_depth(self.depth_image, self.camera.near_plane, self.camera.far_plane)

  @cached_property
  def median_ndc_image(self) -> torch.Tensor:
    return ndc_depth(self.median_depth_image, self.camera.near_plane, self.camera.far_plane)
  
  @property
  def image_size(self) -> Tuple[int, int]:
    return self.camera.image_size
  
  
  def detach(self):
    return Rendering(
      **{k: x.detach() if hasattr(x, 'detach') else x
          for k, x in unpack(self).items()})

