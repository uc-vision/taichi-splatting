from dataclasses import replace
from functools import cache
from typing import Tuple
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.concurrent import warp_add_vector_32, warp_add_vector_64


@cache
def forward_kernel(config: RasterConfig, feature_size: int, dtype=ti.f32):
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D

  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf

  @ti.kernel
  def _forward_kernel(
      # Input tensors
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),              # [N, 7] 2D gaussian parameters
      point_features: ti.types.ndarray(feature_vec, ndim=1),         # [N, F] gaussian features

      # Tile data structures
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),  # [T] start/end range of overlapping points
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),            # [P] mapping from overlap index to point index

      # Output image buffers
      image_feature: ti.types.ndarray(feature_vec, ndim=2),          # [H, W, F] output features
      image_alpha: ti.types.ndarray(dtype, ndim=2),                 # [H, W] output alpha
  ):
    camera_height, camera_width = image_feature.shape
    tiles_wide = (camera_width + tile_size - 1) // tile_size
    tiles_high = (camera_height + tile_size - 1) // tile_size

    ti.loop_config(block_dim=(tile_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):
      pixel = tiling.tile_transform(tile_id, tile_idx, tile_size, (1, 1), tiles_wide)
      pixelf = ti.cast(pixel, dtype) + 0.5

      # Initialize accumulators for all pixels in tile
      in_bounds = pixel.y < camera_height and pixel.x < camera_width

      accum_features = feature_vec(0.0)
      total_weight = 0.0 if in_bounds else 1.0

      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset
      num_point_groups = tiling.round_up(tile_point_count, tile_area)

      # Open shared memory arrays
      tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
      tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)
      tile_point_id = ti.simt.block.SharedArray((tile_area, ), dtype=ti.i32)

      for point_group_id in range(num_point_groups):
        if ti.simt.block.sync_all_nonzero(ti.i32(total_weight >= ti.static(config.saturate_threshold))):
          break

        # Load points into shared memory
        group_start_offset = start_offset + point_group_id * tile_area
        load_index = group_start_offset + tile_idx

        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]
          tile_point[tile_idx] = points[point_idx]
          tile_feature[tile_idx] = point_features[point_idx]
          tile_point_id[tile_idx] = point_idx
        ti.simt.block.sync()

        remaining_points = tile_point_count - point_group_id

        # Process all points in group for each pixel in tile
        for in_group_idx in range(min(tile_area, remaining_points)):
          if ti.simt.warp.all_nonzero(ti.u32(0xffffffff), 
              ti.i32(total_weight >= ti.static(config.saturate_threshold))):
            break

          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

          gaussian_alpha = pdf(pixelf, mean, axis, sigma)
          alpha = point_alpha * gaussian_alpha
          alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

          if alpha > config.alpha_threshold:
            weight = alpha * (1.0 - total_weight)

            accum_features += tile_feature[in_group_idx] * weight
            total_weight += weight

      # Write final results
      if in_bounds:
        image_feature[pixel.y, pixel.x] = accum_features
        image_alpha[pixel.y, pixel.x] = total_weight

  return _forward_kernel


@cache
def query_visibility_kernel(config: RasterConfig, dtype=ti.f32):
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D
  vec1 = lib.vec1

  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf
  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64


  @ti.kernel
  def _query_visibility_kernel(
      # Input tensors
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),              # [N, 7] 2D gaussian parameters

      # Tile data structures  
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),  # [T] start/end range of overlapping points
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),            # [P] mapping from overlap index to point index

      # Output buffers
      point_visibility: ti.types.ndarray(dtype, ndim=1),            # [N] visibility per point
      image_size: ti.math.ivec2                                     # (W, H) image dimensions
  ):
    camera_width, camera_height = image_size
    tiles_wide = (camera_width + tile_size - 1) // tile_size
    tiles_high = (camera_height + tile_size - 1) // tile_size

    ti.loop_config(block_dim=(tile_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):
      pixel = tiling.tile_transform(tile_id, tile_idx, tile_size, (1, 1), tiles_wide)
      pixelf = ti.cast(pixel, dtype) + 0.5

      # Initialize accumulators for all pixels in tile
      in_bounds = pixel.y < camera_height and pixel.x < camera_width
      total_weight = 0.0 if in_bounds else 1.0

      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset
      num_point_groups = tiling.round_up(tile_point_count, tile_area)

      # Open shared memory arrays
      tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
      tile_point_id = ti.simt.block.SharedArray((tile_area, ), dtype=ti.i32)
      tile_visibility = ti.simt.block.SharedArray((tile_area, ), dtype=vec1)
      

      for point_group_id in range(num_point_groups):
        if ti.simt.block.sync_all_nonzero(ti.i32(total_weight >= ti.static(config.saturate_threshold))):
          break

        # Load points into shared memory
        group_start_offset = start_offset + point_group_id * tile_area
        load_index = group_start_offset + tile_idx

        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]
          tile_point[tile_idx] = points[point_idx]
          tile_point_id[tile_idx] = point_idx
          tile_visibility[tile_idx] = 0.0
          
        ti.simt.block.sync()

        remaining_points = tile_point_count - point_group_id

        # Process all points in group for each pixel in tile
        for in_group_idx in range(min(tile_area, remaining_points)):
          if ti.simt.warp.all_nonzero(ti.u32(0xffffffff), 
              ti.i32(total_weight >= ti.static(config.saturate_threshold))):
            break

          gaussian_weight = vec1(0.0)
          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

          gaussian_alpha = pdf(pixelf, mean, axis, sigma)
          alpha = point_alpha * gaussian_alpha
          alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

          if alpha > config.alpha_threshold:
            weight = alpha * (1.0 - total_weight)

            gaussian_weight += weight
            total_weight += weight

          if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(gaussian_weight[0] > 0.0)):
            # Accumulate visibility in shared memory across the warp
            warp_add_vector(tile_visibility[in_group_idx], gaussian_weight)

        ti.simt.block.sync()

        # Write visibility to global memory
        if load_index < end_offset and tile_visibility[tile_idx][0] > 0.0:
          point_idx = tile_point_id[tile_idx]

          if ti.static(config.compute_visibility):
            ti.atomic_add(point_visibility[point_idx], tile_visibility[tile_idx])

  return _query_visibility_kernel
