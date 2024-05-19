import math
from numbers import Integral
from beartype.typing import Tuple
import taichi as ti
from taichi.math import ivec2
import torch
from taichi_splatting import cuda_lib

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.mapper.grid_query import GridQuery
from taichi_splatting.taichi_lib.conversions import torch_taichi

def pad_to_tile(image_size: Tuple[Integral, Integral], tile_size: int):
  def pad(x):
    return int(math.ceil(x / tile_size) * tile_size)
 
  return tuple(pad(x) for x in image_size)


@ti.func
def norm_depth(depth: ti.f32, near:ti.f32, far:ti.f32) -> ti.f32:
  ndc_depth =  (far + near - (2.0 * near * far) / depth) / (far - near)
  return ti.math.clamp((ndc_depth + 1.) / 2., 0.0, 1.0)

  
def make_tile_mapper(grid_query:GridQuery, config:RasterConfig):

  make_query = grid_query.make_query  
  primitive_type = grid_query.primitive
  tile_size = config.tile_size

  if not config.depth16:
    max_tile = 65535
    key_type = torch.uint64
    end_sort_bit = 48

    @ti.func
    def make_sort_key(depth, tile_id):

        depth_32 = ti.bit_cast(depth, ti.u32)
        return ti.cast(depth_32, ti.u64) | (ti.cast(tile_id, ti.u64) << 32)
  
    @ti.func
    def get_tile_id(key):
      return ti.cast(key >> 32, ti.i32)


  else:
    max_tile = 65535
    key_type = torch.uint32
    end_sort_bit = 32

    @ti.func
    def make_sort_key(depth:ti.f32, tile_id:ti.i32):
        return ti.cast(depth * 65535, ti.u32) | (ti.cast(tile_id, ti.u32) << 16)
        
  
    @ti.func
    def get_tile_id(key:ti.u32):
      return ti.cast(key >> 16, ti.i32)



  @ti.kernel
  def tile_overlaps_kernel(
      gaussians: ti.types.ndarray(primitive_type.vec, ndim=1),  
      image_size: ivec2,

      # outputs
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):
      ti.loop_config(block_dim=128)
      for idx in range(gaussians.shape[0]):
          query = make_query(gaussians[idx], image_size)
          counts[idx] =  query.count_tiles()



  @ti.kernel
  def find_ranges_kernel(
      sorted_keys: ti.types.ndarray(torch_taichi[key_type], ndim=1),  # (M)
      # output tile_ranges (tile id -> start, end)
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),   
  ):  
    ti.loop_config(block_dim=1024)
    for idx in range(sorted_keys.shape[0]):
        # tile id is in the 32 high bits of the 64 bit key
        tile_id = get_tile_id(sorted_keys[idx])
        
        next_tile_id = max_tile
        if idx + 1 < sorted_keys.shape[0]:
          next_tile_id = get_tile_id(sorted_keys[idx + 1])

        
        if tile_id != next_tile_id:
            tile_ranges[tile_id][1] = idx + 1

            if next_tile_id < max_tile:
              tile_ranges[next_tile_id][0] = idx + 1


  @ti.kernel
  def generate_sort_keys_kernel(
      valid_idx:ti.types.ndarray(ti.i64, ndim=1),
      # tile_counts:ti.types.ndarray(ti.i32, ndim=1),

      depths: ti.types.ndarray(ti.f32, ndim=1),  # (M)
      near:ti.f32, far:ti.f32,

      gaussians : ti.types.ndarray(primitive_type.vec, ndim=1),  # (M)
      cumulative_overlap_counts: ti.types.ndarray(ti.i32, ndim=1),  # (M)
      # (K), K = sum(num_overlap_tiles)
      image_size: ivec2,

      # outputs
      overlap_sort_key: ti.types.ndarray(torch_taichi[key_type], ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):
    tiles_wide = image_size.x // tile_size

    ti.loop_config(block_dim=128)
    for i in range(valid_idx.shape[0]):
      idx = valid_idx[i]
      
      query = make_query(gaussians[idx], image_size)
      key_idx = cumulative_overlap_counts[idx]
      
      depth = norm_depth(depths[idx], near, far)
      # depth = depths[idx]
      # n = query.count_tiles()
      # if n != tile_counts[idx]:
      #   print(f"{idx}: Tile count mismatch {n} != {tile_counts[idx]}")

      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile
          tile_id = tile.x + tile.y * tiles_wide
      
          key = make_sort_key(depth, tile_id)

          # sort based on tile_id, depth
          overlap_sort_key[key_idx] = key
          overlap_to_point[key_idx] = ti.cast(idx, ti.i32) # map overlap index back to point index
          key_idx += 1


  def sort_tile_depths(valid_idx:torch.Tensor,
                       tile_counts:torch.Tensor,
                       depths:torch.Tensor, depth_range:tuple[float, float], 
                      tile_overlap_ranges:torch.Tensor, 
                      cum_overlap_counts:torch.Tensor, 
                      total_overlap:int, image_size):
    
    # assert tile_counts[valid_idx].sum() == total_overlap, "Invalid tile counts"
    # assert (tile_counts[valid_idx] > 0).all(), "Invalid tile counts"

    overlap_key = torch.empty((total_overlap, ), dtype=key_type, device=cum_overlap_counts.device)
    overlap_to_point = torch.empty((total_overlap, ), dtype=torch.int32, device=cum_overlap_counts.device)

    near, far  = depth_range
    generate_sort_keys_kernel(valid_idx, depths.contiguous(), near, far, tile_overlap_ranges, cum_overlap_counts, image_size,
                              overlap_key, overlap_to_point)

    overlap_key, overlap_to_point  = cuda_lib.radix_sort_pairs(overlap_key, overlap_to_point, end_bit=end_sort_bit)
    return overlap_key, overlap_to_point
  

  # def generate_tile_overlaps(gaussians, image_size):
  #   overlap_counts = torch.empty( (gaussians.shape[0], ), dtype=torch.int32, device=gaussians.device)

  #   tile_overlaps_kernel(gaussians, ivec2(image_size), overlap_counts)    
          
  #   cum_overlap_counts, total_overlap = cuda_lib.full_cumsum(overlap_counts)
  #   return cum_overlap_counts[:-1], total_overlap

  def f(gaussians : torch.Tensor, 
        depths:torch.Tensor, 
        depth_range:tuple[float, float], 
        
        tile_counts:torch.Tensor,
        image_size:Tuple[Integral, Integral]):

    image_size = pad_to_tile(image_size, tile_size)
    tile_shape = (image_size[1] // tile_size, image_size[0] // tile_size)


    assert tile_shape[0] * tile_shape[1] < max_tile, \
      f"tile dimensions {tile_shape} for image size {image_size} exceed maximum tile count (16 bit id), try increasing tile_size" 


    with torch.no_grad():
      valid_idx = torch.nonzero(tile_counts).squeeze(1)

      cum_overlap_counts, total_overlap = cuda_lib.full_cumsum(tile_counts)
      cum_overlap_counts = cum_overlap_counts[:-1]

      # This needs to be initialised to zeros (not empty)
      # as sometimes there are no overlaps for a tile
      tile_ranges = torch.zeros((*tile_shape, 2), dtype=torch.int32, device=gaussians.device)

      if total_overlap > 0:
        overlap_key, overlap_to_point = sort_tile_depths(valid_idx, tile_counts,
          depths, depth_range, gaussians, cum_overlap_counts, total_overlap, image_size)
        
        find_ranges_kernel(overlap_key, tile_ranges.view(-1, 2))
      else:
        overlap_to_point = torch.empty((0, ), dtype=torch.int32, device=gaussians.device)


      # assert (overlap_to_point < gaussians.shape[0]).all(), "Invalid overlap_to_point index"
      # assert (tile_ranges[..., 0] <= tile_ranges[..., 1]).all(), "Invalid tile ranges"
      # assert (tile_ranges <= total_overlap).all(), "Invalid tile ranges"

      return overlap_to_point, tile_ranges
      
  return f


