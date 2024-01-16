from functools import cache
import math
from numbers import Integral
from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2
import torch
from taichi_splatting import cuda_lib
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.taichi_lib.f32 import (Gaussian2D)


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
  def tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),  
      image_size: ivec2,

      # outputs
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):
      for idx in range(gaussians.shape[0]):
          query = grid_query(gaussians[idx], image_size)
          counts[idx] =  query.count_tiles()

  @ti.kernel
  def count_tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1), 
      image_size: ivec2,
      tile_counts: ti.types.ndarray(ti.i32, ndim=2), 
  ):
      for idx in range(gaussians.shape[0]):
          lower, upper = grid_ops.gaussian_tile_ranges(gaussians[idx], image_size)
          for tile_uv in ti.grouped(ti.ndrange((lower.x, upper.x), (lower.y, upper.y))):
              tile_counts[tile_uv] += 1




  def count_tile_overlaps(gaussians:torch.Tensor, image_size:Tuple[Integral, Integral]):
    tile_counts = torch.zeros((image_size[0] // tile_size, image_size[1] // tile_size), 
                              dtype=torch.int32, device=gaussians.device)
    
    count_tile_overlaps_kernel(gaussians, ivec2(image_size), tile_counts)
    return tile_counts



  @ti.kernel
  def find_ranges_kernel(
      sorted_keys: ti.types.ndarray(ti.i64, ndim=1),  # (M)
      # output tile_ranges (tile id -> start, end)
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),   
  ):  
    for idx in range(sorted_keys.shape[0]):
        # tile id is in the 32 high bits of the 64 bit key
        tile_id = sorted_keys[idx] >> 32

        if idx < sorted_keys.shape[0] - 1:
          next_tile_id = sorted_keys[idx + 1] >> 32
          
          if tile_id != next_tile_id:
              tile_ranges[next_tile_id][0] = idx + 1
              tile_ranges[tile_id][1] = idx + 1
        else:
          tile_ranges[tile_id][1] = idx + 1



  @ti.kernel
  def generate_sort_keys_kernel(
      depth: ti.types.ndarray(ti.i32, ndim=1),  # (M)
      gaussians : ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M)
      cumulative_overlap_counts: ti.types.ndarray(ti.i32, ndim=1),  # (M)
      # (K), K = sum(num_overlap_tiles)
      image_size: ivec2,

      # outputs
      overlap_sort_key: ti.types.ndarray(ti.i64, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):
    tiles_wide = image_size.x // tile_size

    for idx in range(cumulative_overlap_counts.shape[0]):
      encoded_depth_key = depth[idx]
      query = grid_query(gaussians[idx], image_size)

      key_idx = cumulative_overlap_counts[idx]

      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile

          encoded_tile_id = ti.cast(tile.x + tile.y * tiles_wide, ti.i32)
          
          sort_key = ti.cast(encoded_depth_key, ti.i64) | (
            ti.cast(encoded_tile_id, ti.i64) << 32)
      
          overlap_sort_key[key_idx] = sort_key # sort based on tile_id, depth
          overlap_to_point[key_idx] = idx # map overlap index back to point index
          key_idx += 1

  def sort_tile_depths(depths:torch.Tensor, tile_overlap_ranges:torch.Tensor, cum_overlap_counts:torch.Tensor, total_overlap:int, image_size):

    overlap_key = torch.empty((total_overlap, ), dtype=torch.int64, device=cum_overlap_counts.device)
    overlap_to_point = torch.empty((total_overlap, ), dtype=torch.int32, device=cum_overlap_counts.device)

    generate_sort_keys_kernel(depths, tile_overlap_ranges, cum_overlap_counts, image_size,
                              overlap_key, overlap_to_point)

    overlap_key, overlap_to_point  = cuda_lib.sort_pairs(overlap_key, overlap_to_point)
    return overlap_key, overlap_to_point
  

  def generate_tile_overlaps(gaussians, image_size):
    overlap_counts = torch.empty( (gaussians.shape[0], ), dtype=torch.int32, device=gaussians.device)

    tile_overlaps_kernel(gaussians, ivec2(image_size), overlap_counts)

    cum_overlap_counts, total_overlap = cuda_lib.full_cumsum(overlap_counts)
    return cum_overlap_counts[:-1], total_overlap


  def f(gaussians : torch.Tensor, depths:torch.Tensor, image_size:Tuple[Integral, Integral]):

    image_size = pad_to_tile(image_size, tile_size)

    with torch.no_grad():
      cum_overlap_counts, total_overlap = generate_tile_overlaps(
        gaussians, image_size)
            
      tile_shape = (image_size[1] // tile_size, image_size[0] // tile_size)

      # This needs to be initialised to zeros (not empty)
      # as sometimes there are no overlaps for a tile
      tile_ranges = torch.zeros((*tile_shape, 2), dtype=torch.int32, device=gaussians.device)

      if total_overlap > 0:
        overlap_key, overlap_to_point = sort_tile_depths(
          depths, gaussians, cum_overlap_counts, total_overlap, image_size)
        find_ranges_kernel(overlap_key, tile_ranges.view(-1, 2))
      else:
        overlap_to_point = torch.empty((0, ), dtype=torch.int32, device=gaussians.device)

      return overlap_to_point, tile_ranges
      
  return f


@beartype
def map_to_tiles(gaussians : torch.Tensor, encoded_depth:torch.Tensor, 
                 image_size:Tuple[Integral, Integral],
                 config:RasterConfig
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps guassians to tiles, sorted by depth (front to back):
    Parameters:
     gaussians: (N, 6) torch.Tensor of packed gaussians, N is the number of gaussians
     encoded_depths: (N)  torch.Tensor of encoded depths (int32)
     image_size: (2, ) tuple of ints, (width, height)
     tile_config: configuration for tile mapper (tile_size etc.)

    Returns:
     overlap_to_point: (K, ) torch tensor, where K is the number of overlaps, maps overlap index to point index
     tile_ranges: (M, 2) torch tensor, where M is the number of tiles, maps tile index to range of overlap indices
    """

  
  mapper = tile_mapper(config)
  return mapper(gaussians, encoded_depth, image_size)