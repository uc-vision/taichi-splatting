from functools import cache

from taichi_splatting.autograd import restore_grad

from .forward import Config, forward_kernel
from .backward import backward_kernel

from numbers import Integral
from typing import Tuple

import torch
from beartype import beartype

@cache
def render_function(config:Config, num_features:int):
  forward = forward_kernel(config, num_features)
  backward = backward_kernel(config, num_features)

  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians: torch.Tensor, features: torch.Tensor,
                overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                image_size: Tuple[Integral, Integral]
                ) -> torch.Tensor:
        
      shape = (image_size[1], image_size[0])
      image_feature = torch.zeros((*shape, features.shape[1]),
                                  dtype=torch.float32, device=features.device)
      image_alpha = torch.zeros(shape, dtype=torch.float32, device=features.device)
      image_last_valid = torch.zeros(shape, dtype=torch.int32, device=features.device)

      forward(gaussians, features, 
        tile_overlap_ranges, overlap_to_point,
        image_feature, image_alpha, image_last_valid)

      # Non differentiable parameters
      ctx.overlap_to_point = overlap_to_point
      ctx.tile_overlap_ranges = tile_overlap_ranges
      ctx.image_last_valid = image_last_valid
      ctx.image_alpha = image_alpha

      ctx.save_for_backward(gaussians, features, 
                            image_feature)
      
      return image_feature

    @staticmethod
    def backward(ctx, grad_image_feature:torch.Tensor):
        gaussians, features, image_feature = ctx.saved_tensors

        grad_gaussians = torch.zeros_like(gaussians)
        grad_features = torch.zeros_like(features)

        with restore_grad(grad_gaussians, grad_features):

          backward(gaussians, features, 
            ctx.tile_overlap_ranges, ctx.overlap_to_point,
            image_feature, ctx.image_alpha, ctx.image_last_valid,
            grad_image_feature,
            grad_gaussians, grad_features)

          return grad_gaussians, grad_features, None, None, None, None
  return _module_function

@beartype
def rasterize(gaussians: torch.Tensor, features: torch.Tensor,
              overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
              image_size: Tuple[Integral, Integral], config: Config
              ) -> torch.Tensor:
  """
  Paraeters:
      gaussians: (N, 6)  packed gaussians, N is the number of gaussians
      features: (N, F)   features, F is the number of features

      tile_overlap_ranges: (TH, TW, 2) M is the number of tiles, 
        maps tile index to range of overlap indices (0..K]
      overlap_to_point: (K, )  K is the number of overlaps, 
        maps overlap index to point index (0..N]

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

    Returns:
      image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
  """
  _module_function = render_function(config, features.shape[1])

  return _module_function.apply(gaussians, features, 
          overlap_to_point, tile_overlap_ranges, 
          image_size)