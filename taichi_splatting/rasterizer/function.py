from functools import cache

from taichi_splatting.autograd import restore_grad
from taichi_splatting.tile_mapper import map_to_tiles, pad_to_tile

from .forward import RasterConfig, forward_kernel
from .backward import backward_kernel

from numbers import Integral
from typing import Tuple

import torch
from beartype import beartype

@cache
def render_function(config:RasterConfig,
                    points_requires_grad:bool,
                    features_requires_grad:bool, 
                    feature_size:int):
    
  forward = forward_kernel(config, feature_size=feature_size)
  backward = backward_kernel(config, points_requires_grad,
                             features_requires_grad, feature_size)

  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians: torch.Tensor, features: torch.Tensor,
                overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                image_size: Tuple[Integral, Integral]
                ) -> torch.Tensor:
        
      shape = (image_size[1], image_size[0])
      image_feature = torch.empty((*shape, features.shape[1]),
                                  dtype=torch.float32, device=features.device)
      image_alpha = torch.empty(shape, dtype=torch.float32, device=features.device)
      image_last_valid = torch.empty(shape, dtype=torch.int32, device=features.device)

      forward(gaussians, features, 
        tile_overlap_ranges, overlap_to_point,
        image_feature, image_alpha, image_last_valid)

      # Non differentiable parameters
      ctx.overlap_to_point = overlap_to_point
      ctx.tile_overlap_ranges = tile_overlap_ranges
      ctx.image_last_valid = image_last_valid
      ctx.image_alpha = image_alpha
      ctx.image_size = image_size

      ctx.mark_non_differentiable(image_alpha, image_last_valid)
      ctx.save_for_backward(gaussians, features, image_feature)
            
      return image_feature, image_alpha

    @staticmethod
    def backward(ctx, grad_image_feature:torch.Tensor, grad_alpha:torch.Tensor):
        gaussians, features, image_feature = ctx.saved_tensors

        grad_gaussians = torch.zeros_like(gaussians)
        grad_features = torch.zeros_like(features)
  
        with restore_grad(grad_gaussians, grad_features):

          backward(gaussians, features, 
            ctx.tile_overlap_ranges, ctx.overlap_to_point,
            image_feature, ctx.image_alpha, ctx.image_last_valid,
            grad_image_feature.contiguous(),
            grad_gaussians, grad_features)

          return grad_gaussians, grad_features, None, None, None, None
  return _module_function





@beartype
def rasterize_with_tiles(gaussians2d: torch.Tensor, features: torch.Tensor,
              overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
              image_size: Tuple[Integral, Integral], config: RasterConfig
              ) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Rasterize an image given 2d gaussians, features and tile overlap information.
  Consider using rasterize instead to also compute tile overlap information.

  Parameters:
      gaussians2d: (N, 6)  packed gaussians, N is the number of gaussians
      features: (N, F)   features, F is the number of features

      tile_overlap_ranges: (TH, TW, 2) M is the number of tiles, 
        maps tile index to range of overlap indices (0..K]
      overlap_to_point: (K, )  K is the number of overlaps, 
        maps overlap index to point index (0..N]

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

    Returns:
      image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
      alpha: (H, W) torch tensor, where H, W are the image height and width
  """
  _module_function = render_function(config, gaussians2d.requires_grad,
                                      features.requires_grad, features.shape[1])

  return _module_function.apply(gaussians2d, features, 
          overlap_to_point, tile_overlap_ranges, 
          image_size)


def rasterize(gaussians2d:torch.Tensor, depths:torch.Tensor, 
                          features:torch.Tensor, image_size:Tuple[Integral, Integral],
                          config:RasterConfig):
    
    
  """
  Rasterize an image given 2d gaussians, features. 

  Parameters:
      gaussians2d: (N, 6)  packed gaussians, N is the number of gaussians
      features: (N, F)   features, F is the number of features

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

    Returns:
      image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
      alpha: (H, W) torch tensor, where H, W are the image height and width
  """

  # render with padding to tile_size, later crop back to original size
  padded_size = pad_to_tile(image_size, config.tile_size)
  overlap_to_point, ranges = map_to_tiles(gaussians2d, depths, 
    image_size=padded_size, config=config)

  image, alpha = rasterize_with_tiles(gaussians2d, features, 
    tile_overlap_ranges=ranges, overlap_to_point=overlap_to_point,
    image_size=image_size, config=config)

  return image, alpha 