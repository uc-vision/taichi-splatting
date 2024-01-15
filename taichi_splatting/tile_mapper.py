from functools import cache
import math
from numbers import Integral
from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2
import torch
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.taichi_lib.f32 import (Gaussian2D)
from taichi_splatting.cuda_lib import full_cumsum

from taichi_splatting.taichi_lib.grid_query import make_grid_query

def pad_to_tile(image_size: Tuple[Integral, Integral], tile_size: int):
  def pad(x):
    return int(math.ceil(x / tile_size) * tile_size)
 
  return tuple(pad(x) for x in image_size)


@cache
def tile_mapper(config:RasterConfig):


  tile_size = config.tile_size
  grid_ops = make_grid_query(
    tile_size=tile_size, 
    gaussian_scale=config.gaussian_scale, 
    tight_culling=config.tight_culling)
  
  grid_query = grid_ops.grid_query
  

  @ti.kernel
  def count_tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1), 
      image_size: ivec2,
      tile_counts: ti.types.ndarray(ti.i32, ndim=2), 
  ):
      for idx in range(gaussians.shape[0]):
          uv, uv_conic, _ = Gaussian2D.unpack(gaussians[idx])

          lower, upper = grid_ops.gaussian_tile_ranges(uv, uv_conic, image_size)
          for tile_uv in ti.grouped(ti.ndrange((lower.x, upper.x), (lower.y, upper.y))):
              tile_counts[tile_uv] += 1


  def count_tile_overlaps(gaussians:torch.Tensor, image_size:Tuple[Integral, Integral]):
    tile_counts = torch.zeros((image_size[0] // tile_size, image_size[1] // tile_size), 
                              dtype=torch.int32, device=gaussians.device)
    
    count_tile_overlaps_kernel(gaussians, ivec2(image_size), tile_counts)
    return tile_counts


  @ti.kernel
  def partition_tiles_kernel(
      depth: ti.types.ndarray(ti.f32, ndim=2),  # (M, >= 1)
      gaussians : ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M)
      tile_offsets: ti.types.ndarray(ti.i32, ndim=2),  # (H, W)

      image_size: ivec2,

      # output - precise tile counts
      tile_counts : ti.types.ndarray(ti.i32, ndim=2),  # (H, W)

      overlap_depths: ti.types.ndarray(ti.f32, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):

    for point_idx in range(gaussians.shape[0]):
      query = grid_query(gaussians[point_idx], image_size)
      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile

          tile_idx = ti.atomic_add(tile_counts[tile], 1)
          key_idx = tile_offsets[tile] + tile_idx

          overlap_depths[key_idx] = depth[point_idx, 0] # sort based on tile_id, depth
          overlap_to_point[key_idx] = point_idx # map overlap index back to point index


  def f(gaussians : torch.Tensor, depths:torch.Tensor, image_size:Tuple[Integral, Integral]):

    image_size = pad_to_tile(image_size, tile_size)

    with torch.no_grad():
      max_overlap_counts = count_tile_overlaps(gaussians, image_size)          
      overlap_sums, max_overlaps = full_cumsum(max_overlap_counts.view(-1))

      overlap_depths = torch.empty((max_overlaps,), dtype=torch.float32, device=gaussians.device)
      overlap_to_point = torch.empty((max_overlaps,), dtype=torch.int32, device=gaussians.device)

      overlap_offsets = overlap_sums[:-1].view(max_overlap_counts.shape)
      overlap_counts = torch.zeros_like(max_overlap_counts)

      partition_tiles_kernel(depths, gaussians, overlap_offsets, 
                             ivec2(image_size), overlap_counts, overlap_depths, overlap_to_point)

      tile_ranges = torch.stack(
        [overlap_offsets, overlap_offsets + overlap_counts], dim=1)



      return overlap_to_point, tile_ranges.view(-1, 2)
      
  return f


@beartype
def map_to_tiles(gaussians : torch.Tensor, depths:torch.Tensor, 
                 image_size:Tuple[Integral, Integral],
                 config:RasterConfig
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps guassians to tiles, sorted by depth (front to back):
    Parameters:
     gaussians: (N, 6) torch.Tensor of packed gaussians, N is the number of gaussians
     depths: (N, 1 or 3)  torch.Tensor of depths or depth + depth variance + depth^2
     image_size: (2, ) tuple of ints, (width, height)
     tile_config: configuration for tile mapper (tile_size etc.)

    Returns:
     overlap_to_point: (K, ) torch tensor, where K is the number of overlaps, maps overlap index to point index
     tile_ranges: (M, 2) torch tensor, where M is the number of tiles, maps tile index to range of overlap indices
    """

  
  mapper = tile_mapper(config)
  return mapper(gaussians, depths, image_size)