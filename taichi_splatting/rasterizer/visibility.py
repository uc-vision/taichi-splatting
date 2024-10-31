from functools import cache
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib.concurrent import WARP_SIZE, block_reduce_i32, warp_add_vector_32, warp_add_vector_64

from taichi_splatting.taichi_lib import get_library
from taichi.math import ivec2




@cache
def visibility_kernel(config: RasterConfig, dtype=ti.f32):
  
  lib = get_library(dtype)
  Gaussian2D, vec2, vec1 = lib.Gaussian2D, lib.vec2, lib.vec1
  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64
  
  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  thread_pixels = config.pixel_stride[0] * config.pixel_stride[1]
  block_area = tile_area // thread_pixels
  
  assert block_area >= WARP_SIZE, \
    f"pixel_stride {config.pixel_stride} and tile_size, {config.tile_size} must allow at least one warp sized ({WARP_SIZE}) tile"

  # each thread is responsible for a small tile of pixels
  pixel_tile = tuple([ (i, 
            (i % config.pixel_stride[0],
            i // config.pixel_stride[0]))
              for i in range(thread_pixels) ])

  # types for each thread to keep state in it's tile of pixels
  thread_vector = ti.types.vector(thread_pixels, dtype=dtype)
  thread_index = ti.types.vector(thread_pixels, dtype=ti.i32)


  gaussian_pdf = lib.gaussian_pdf_antialias_with_grad if config.antialias else lib.gaussian_pdf_with_grad


  @ti.kernel
  def _visibility_kernel(
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, 6)
      
      # (TH, TW, 2) the start/end (0..K] index of ranges in the overlap_to_point array
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      # (K) ranges of points mapping to indexes into points list
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
      
      # saved from forward
      image_alpha: ti.types.ndarray(dtype, ndim=2),       # H, W
      image_last_valid: ti.types.ndarray(ti.i32, ndim=2),  # H, W

      # output gradients
      visibility: ti.types.ndarray(ti.i32, ndim=1),  # (M)
  ):

    camera_height, camera_width = image_alpha.shape

    # round up
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size

    # see forward.py for explanation of tile_id and tile_idx and blocking
    ti.loop_config(block_dim=(block_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, block_area):
      pixel_base = tiling.tile_transform(tile_id, tile_idx, 
                        tile_size, config.pixel_stride, tiles_wide)

      # open the shared memory
      tile_point_id = ti.simt.block.SharedArray((block_area, ), dtype=ti.i32)
      tile_point = ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)

      tile_visibility = ti.simt.block.SharedArray((block_area,), dtype=vec1) 

      
      last_point_pixel = thread_index(0)
      T_i = thread_vector(1.0)

      for i, offset in ti.static(pixel_tile):
        pixel = ivec2(offset) + pixel_base

        if pixel.y < camera_height and pixel.x < camera_width:
          last_point_pixel[i] = image_last_valid[pixel.y, pixel.x]
          T_i[i] = 1.0 - image_alpha[pixel.y, pixel.x]

      last_point_thread = last_point_pixel.max()

      # fine tune the end offset to the actual number of points renderered
      end_offset = block_reduce_i32(last_point_thread, ti.max, ti.atomic_max, 0)

      start_offset, _ = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset

      num_point_groups = (tile_point_count + ti.static(block_area - 1)) // block_area

      # Loop through the range in groups of block_area
      for point_group_id in range(num_point_groups):
        ti.simt.block.sync() 
        
        # load points and features into block shared memory
        group_offset_base = point_group_id * block_area

        block_end_idx = end_offset - group_offset_base
        block_start_idx = ti.max(block_end_idx - block_area, 0)

        load_index = block_end_idx - tile_idx - 1
        if load_index >= block_start_idx:
          point_idx = overlap_to_point[load_index]

          tile_point_id[tile_idx] = point_idx
          tile_point[tile_idx] = points[point_idx]
          
          tile_visibility[tile_idx] = vec1(0.0)

        point_group_size = ti.min(
          block_area, tile_point_count - group_offset_base)
                    
        ti.simt.block.sync()

        for in_group_idx in range(point_group_size):
          point_index = end_offset - (group_offset_base + in_group_idx)

          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])
          gaussian_visibility = vec1(0.0)

          has_grad = False
          for i, offset in ti.static(pixel_tile):
            pixelf = ti.cast(pixel_base, dtype) + vec2(offset) + 0.5
            gaussian_alpha, _, _, _ = gaussian_pdf(pixelf, mean, axis, sigma)
            
            alpha = point_alpha * gaussian_alpha
            pixel_grad = (alpha >= ti.static(config.alpha_threshold)) and (point_index <= last_point_pixel[i])      
      
            if pixel_grad:
              has_grad = True
              
              alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))
              T_i[i] /= (1. - alpha)

              gaussian_visibility += vec1(alpha * T_i[i])      


          if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(has_grad)):
            warp_add_vector(tile_visibility[in_group_idx], gaussian_visibility)
        # end of point group loop

        ti.simt.block.sync()

        # finally accumulate gradients in global memory
        if load_index >= block_start_idx:
          point_offset = tile_point_id[tile_idx] 
          ti.atomic_add(visibility[point_offset], tile_visibility[tile_idx])

      # end of point group id loop
    # end of pixel loop

  return _visibility_kernel




