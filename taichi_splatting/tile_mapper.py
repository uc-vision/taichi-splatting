from functools import cache
import math
from numbers import Integral
from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2
import torch
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.taichi_lib.f32 import (Gaussian2D, vec3)


from taichi_splatting.taichi_lib.grid_query import make_grid_query

def pad_to_tile(image_size: Tuple[Integral, Integral], tile_size: int):
  def pad(x):
    return int(math.ceil(x / tile_size) * tile_size)
 
  return tuple(pad(x) for x in image_size)



@cache
def tile_mapper(config:RasterConfig):

  tile_size = config.tile_size
  grid_query = make_grid_query(
    tile_size=tile_size, 
    gaussian_scale=config.gaussian_scale, 
    tight_culling=config.tight_culling)
  

  @ti.kernel
  def tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),  
      image_size: ivec2,

      # outputs
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):
      counts[0] = 0
      for idx in range(gaussians.shape[0]):
          query = grid_query(gaussians[idx], image_size)
          counts[idx + 1] =  query.count_tiles()


  @ti.kernel
  def find_ranges_kernel(
      sorted_keys: ti.types.ndarray(ti.i64, ndim=1),  # (M)
      # output tile_ranges (tile id -> start, end)
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),   
  ):  
    for idx in range(sorted_keys.shape[0] - 1):
        # tile id is in the 32 high bits of the 64 bit key
        tile_id = ti.cast(sorted_keys[idx] >> 32, ti.i32)
        next_tile_id = ti.cast(sorted_keys[idx + 1] >> 32, ti.i32)

        if tile_id != next_tile_id:
            tile_ranges[next_tile_id][0] = idx + 1
            tile_ranges[tile_id][1] = idx + 1

    last_tile_id = ti.cast(sorted_keys[sorted_keys.shape[0] - 1] >> 32, ti.i32)
    tile_ranges[last_tile_id][1] = sorted_keys.shape[0]


  @ti.kernel
  def generate_sort_keys_kernel(
      depth: ti.types.ndarray(ti.f32, ndim=2),  # (M, >= 1)
      gaussians : ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M)
      cumulative_overlap_counts: ti.types.ndarray(ti.i64, ndim=1),  # (M)
      # (K), K = sum(num_overlap_tiles)
      image_size: ivec2,

      # outputs
      overlap_sort_key: ti.types.ndarray(ti.i64, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):
    tiles_wide = image_size.x // tile_size

    for idx in range(cumulative_overlap_counts.shape[0]):
      encoded_depth_key = ti.bit_cast(depth[idx, 0], ti.i32)
      query = grid_query(gaussians[idx], image_size)

      overlap_idx = 0
      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile

          key_idx = cumulative_overlap_counts[idx] + overlap_idx
          encoded_tile_id = ti.cast(tile.x + tile.y * tiles_wide, ti.i32)
          
          sort_key = ti.cast(encoded_depth_key, ti.i64) + (
            ti.cast(encoded_tile_id, ti.i64) << 32)
      
          overlap_sort_key[key_idx] = sort_key # sort based on tile_id, depth
          overlap_to_point[key_idx] = idx # map overlap index back to point index
          overlap_idx += 1

  def sort_tile_depths(depths:torch.Tensor, tile_overlap_ranges:torch.Tensor, cum_overlap_counts:torch.Tensor, total_overlap:int, image_size):

    overlap_key = torch.empty((total_overlap, ), dtype=torch.int64, device=cum_overlap_counts.device)
    overlap_to_point = torch.empty((total_overlap, ), dtype=torch.int32, device=cum_overlap_counts.device)

    generate_sort_keys_kernel(depths, tile_overlap_ranges, cum_overlap_counts, image_size,
                              overlap_key, overlap_to_point)
    

    overlap_key, permutation = torch.sort(overlap_key)
    overlap_to_point = overlap_to_point[permutation]
    return overlap_key, overlap_to_point
  


  def generate_tile_overlaps(gaussians, image_size):
    overlap_counts = torch.empty( (1 + gaussians.shape[0], ), dtype=torch.int32, device=gaussians.device)

    tile_overlaps_kernel(gaussians, ivec2(image_size), overlap_counts)

    cum_overlap_counts = overlap_counts.cumsum(dim=0)
    total_overlap = cum_overlap_counts[-1].item()
    

    return cum_overlap_counts[:-1], total_overlap


  def f(gaussians : torch.Tensor, depths:torch.Tensor, image_size:Tuple[Integral, Integral]):

    with torch.no_grad():
      cum_overlap_counts, total_overlap = generate_tile_overlaps(
        gaussians, image_size)
      
      num_tiles = (image_size[0] // tile_size) * (image_size[1] // tile_size)

      # This needs to be initialised to zeros (not empty)
      # as sometimes there are no overlaps for a tile
      tile_ranges = torch.zeros((num_tiles, 2), dtype=torch.int32, device=gaussians.device)

      if total_overlap > 0:
        overlap_key, overlap_to_point = sort_tile_depths(
          depths, gaussians, cum_overlap_counts, total_overlap, image_size)


        find_ranges_kernel(overlap_key, tile_ranges)
      else:
        overlap_to_point = torch.empty((0, ), dtype=torch.int32, device=gaussians.device)

      return overlap_to_point, tile_ranges
      
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
  w, h = image_size
  assert w % config.tile_size == 0 and h % config.tile_size == 0,\
      f"image size ({w}x{h}) is not divisible by tile size {config.tile_size}"
  
  mapper = tile_mapper(config)
  return mapper(gaussians, depths, image_size)