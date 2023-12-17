from functools import cache
from typing import Tuple
import taichi as ti
from taichi.math import ivec2 

import torch

from gaussian_rasterizer.data_types import Gaussian2D
from gaussian_rasterizer.taichi.projection import radii_from_conic


from taichi.math import ivec4


@cache
def rasterize(feature_type, tile_size:int=16, depth_sort_scale:float=1000.):

  @ti.func
  def gaussian_tile_ranges(
      gaussian: ti.template(),
      image_size: ti.math.ivec2,
  ) -> ivec4:
      uv = gaussian.uv

      # avoid zero radii, at least 1 pixel
      radius = ti.max(radii_from_conic(gaussian.uv_conic), 1.0)  

      min_bound = ti.max(0.0, uv - radius)
      max_bound = uv + radius

      max_tile = image_size // tile_size

      min_tile_bound = ti.cast(min_bound // tile_size, ti.i32)
      min_tile_bound = ti.min(min_tile_bound, max_tile)

      max_tile_bound = ti.cast(max_bound // tile_size, ti.i32) + 1
      max_tile_bound = ti.min(ti.max(max_tile_bound, min_tile_bound + 1),
                          max_tile)

      return min_tile_bound, max_tile_bound
  

  @ti.kernel
  def tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),  
      image_size: ivec2,

      # outputs
      tile_ranges: ti.types.ndarray(ivec4, ndim=1),
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):

      for idx in range(gaussians.shape[0]):
          gaussian = Gaussian2D.unpack(gaussians[idx])

          min_bound, max_bound = gaussian_tile_ranges(gaussian, image_size)
          tile_ranges[idx] = ivec4(*min_bound, *max_bound)

          size = max_bound - min_bound
          counts[idx + 1] = size.x * size.y


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

  def find_tile_ranges(sorted_keys:torch.Tensor):
    tile_ranges = torch.empty((sorted_keys.shape[0], 2), dtype=torch.int32, device=sorted_keys.device)
    find_ranges_kernel(sorted_keys, tile_ranges)
    return tile_ranges

  @ti.kernel
  def generate_sort_keys_kernel(
      depth: ti.types.ndarray(ti.f32, ndim=1),  # (M)
      tile_overlap_ranges : ti.types.ndarray(ivec4, ndim=1), # (M, 4)
      cumulative_overlap_counts: ti.types.ndarray(ti.i64, ndim=1),  # (M)
      # (K), K = sum(num_overlap_tiles)
      image_size: ivec2,

      # outputs
      overlap_sort_key: ti.types.ndarray(ti.i64, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):
    tiles_wide = image_size.x // tile_size

    for idx in range(cumulative_overlap_counts.shape[0]):
      depth_key = ti.cast(depth[idx] * depth_sort_scale, ti.i32)
      
      tile_range = tile_overlap_ranges[idx]
      tile_span = tile_range.zw - tile_range.xy
      
      for tile_u, tile_v in ti.ndrange(tile_span.x, tile_span.y):
        overlap_idx = tile_u + tile_v * tile_span.x # index in overlap list

        key_idx = cumulative_overlap_counts[idx] + overlap_idx
        encoded_tile_id = ti.cast(tile_u + tile_v * tiles_wide, ti.i32)
        
        sort_key = ti.cast(depth_key, ti.i64) + (
           ti.cast(encoded_tile_id, ti.i64) << 32)
    
        overlap_sort_key[key_idx] = sort_key # sort based on tile_id, depth
        overlap_to_point[key_idx] = idx # map overlap index back to point index

  def sort_tile_depths(depths:torch.Tensor, tile_overlap_ranges:torch.Tensor, cum_overlap_counts:torch.Tensor, total_overlap:int, image_size):

    overlap_key = torch.empty((total_overlap, ), dtype=torch.int64, device=cum_overlap_counts.device)
    overlap_to_point = torch.empty((total_overlap, ), dtype=torch.int32, device=cum_overlap_counts.device)

    generate_sort_keys_kernel(depths, tile_overlap_ranges, cum_overlap_counts, image_size,
                              overlap_key, overlap_to_point)

    overlap_key, permutation = torch.sort(overlap_key)
    overlap_to_point = overlap_to_point[permutation]
    return overlap_key, overlap_to_point
  


  def generate_tile_overlaps(gaussians, image_size):
    tile_overlap_ranges = torch.zeros( (gaussians.shape[0], 4), dtype=torch.int32,  device=gaussians.device)
    overlap_counts = torch.zeros( (1 + gaussians.shape[0], ), dtype=torch.int32, device=gaussians.device)

    tile_overlaps_kernel(gaussians, ivec2(image_size), tile_overlap_ranges, overlap_counts)

    cum_overlap_counts = overlap_counts.cumsum(dim=0)
    total_overlap = cum_overlap_counts[-1].item()
    
    print(cum_overlap_counts[:100])


    return cum_overlap_counts, tile_overlap_ranges, total_overlap


  def f(gaussians : torch.Tensor, depths:torch.Tensor, features:torch.Tensor, image_size:Tuple[int, int]):
    
    cum_overlap_counts, tile_overlap_ranges, total_overlap = generate_tile_overlaps(
       gaussians, image_size)
  
    overlap_key, overlap_to_point = sort_tile_depths(
       depths, tile_overlap_ranges, cum_overlap_counts, total_overlap, image_size)
    
    tile_ranges = find_tile_ranges(overlap_key)
    

  return f