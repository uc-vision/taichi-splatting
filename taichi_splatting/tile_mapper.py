from functools import cache
from numbers import Integral
from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2, mat2

import torch

from taichi_splatting.data_types import Gaussian2D
from taichi_splatting.ti.covariance import conic_to_cov, cov_inv_basis, radii_from_cov
from taichi_splatting.ti.bounding import separates_bbox

from taichi.math import ivec4, vec2





@cache
def tile_mapper(tile_size:int=16, gaussian_scale:float=3.0):


  @ti.dataclass
  class GridQuery:
    inv_basis: mat2
    rel_min_bound: vec2

    min_tile: ivec2
    tile_span: ivec2

    @ti.func
    def test_tile(self, tile_uv: ivec2):
      lower = self.rel_min_bound + tile_uv * tile_size
      return not separates_bbox(self.inv_basis, lower, lower + tile_size)
      
  @ti.func 
  def grid_query(v: Gaussian2D.vec, image_size:ivec2) -> GridQuery:
      uv, uv_conic, _ = Gaussian2D.unpack(v)
      uv_cov = conic_to_cov(uv_conic)

      min_tile, max_tile = gaussian_tile_ranges(uv, uv_cov, image_size)
      return GridQuery(
        # Find tiles which intersect the oriented box
        inv_basis = cov_inv_basis(uv_cov, gaussian_scale),
        rel_min_bound = min_tile * tile_size - uv,

        min_tile = min_tile,
        tile_span = max_tile - min_tile)



  @ti.func
  def gaussian_tile_ranges(
      uv: vec2,
      uv_cov: mat2,
      image_size: ti.math.ivec2,
  ) -> ivec4:

      # avoid zero radii, at least 1 pixel
      radius = ti.max(radii_from_cov(uv_cov) * gaussian_scale, 1.0)  

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
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):

      for idx in range(gaussians.shape[0]):
          query = grid_query(gaussians[idx], image_size)
          inside = 0
          for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
            if query.test_tile(tile_uv):
              inside += 1

          counts[idx + 1] = inside



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
      depth: ti.types.ndarray(ti.f32, ndim=1),  # (M)
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
      encoded_depth_key = ti.bit_cast(depth[idx], ti.i32)
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

    overlap_key = torch.zeros((total_overlap, ), dtype=torch.int64, device=cum_overlap_counts.device)
    overlap_to_point = torch.zeros((total_overlap, ), dtype=torch.int32, device=cum_overlap_counts.device)

    generate_sort_keys_kernel(depths, tile_overlap_ranges, cum_overlap_counts, image_size,
                              overlap_key, overlap_to_point)
    

    overlap_key, permutation = torch.sort(overlap_key)
    overlap_to_point = overlap_to_point[permutation]
    return overlap_key, overlap_to_point
  


  def generate_tile_overlaps(gaussians, image_size):
    overlap_counts = torch.zeros( (1 + gaussians.shape[0], ), dtype=torch.int32, device=gaussians.device)

    tile_overlaps_kernel(gaussians, ivec2(image_size), overlap_counts)

    cum_overlap_counts = overlap_counts.cumsum(dim=0)
    total_overlap = cum_overlap_counts[-1].item()
    

    return cum_overlap_counts[:-1], total_overlap


  def f(gaussians : torch.Tensor, depths:torch.Tensor, image_size:Tuple[Integral, Integral]):
    cum_overlap_counts, total_overlap = generate_tile_overlaps(
       gaussians, image_size)
    
    num_tiles = (image_size[0] // tile_size) * (image_size[1] // tile_size)
    tile_ranges = torch.zeros((num_tiles, 2), dtype=torch.int32, device=gaussians.device)

    if total_overlap > 0:
      overlap_key, overlap_to_point = sort_tile_depths(
        depths, gaussians, cum_overlap_counts, total_overlap, image_size)


      find_ranges_kernel(overlap_key, tile_ranges)
    else:
      overlap_to_point = torch.zeros((0, ), dtype=torch.int32, device=gaussians.device)

    return overlap_to_point, tile_ranges
    
  return f

@beartype
def map_to_tiles(gaussians : torch.Tensor, depths:torch.Tensor, 
                 image_size:Tuple[Integral, Integral], tile_size:int=16
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps guassians to tiles, sorted by depth (front to back):
    Parameters:
     gaussians: (N, 6) torch tensor of packed gaussians, N is the number of gaussians
     depths: (N, ) torch float tensor, where N is the number of gaussians
     image_size: (2, ) tuple of ints, (width, height)
     tile_size: int, tile size in pixels

    Returns:
     overlap_to_point: (K, ) torch tensor, where K is the number of overlaps, maps overlap index to point index
     tile_ranges: (M, 2) torch tensor, where M is the number of tiles, maps tile index to range of overlap indices
    """
  
  mapper = tile_mapper(tile_size=tile_size)
  return mapper(gaussians, depths, image_size)