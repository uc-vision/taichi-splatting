
from functools import cache
from numbers import Integral
from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2 

import torch

from gaussian_rasterizer.data_types import Gaussian2D
from taichi.math import ivec4


def rasterize(gaussians : torch.Tensor, features: torch.Tensor, 
            image_size: Tuple[Integral, Integral], tile_size: int
            ) -> torch.Tensor:
  """
      - gaussians: (N, 6)  packed gaussians, N is the number of gaussians
      - features: (N, F)   features, F is the number of features

      - overlap_to_point: (K, )  K is the number of overlaps, maps overlap index to point index (0..N]
      - tile_ranges: (TH, TW, 2) M is the number of tiles, maps tile index to range of overlap indices (0..K]
      
      - image_size: (2, ) tuple of ints, (width, height)
      - tile_size: int, tile size in pixels

    returns:
      - image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
  """

  pass