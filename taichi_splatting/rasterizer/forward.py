
from dataclasses import dataclass
from functools import cache
import taichi as ti
from taichi.math import ivec2


from taichi_splatting.data_types import Gaussian2D
from taichi_splatting.ti.covariance import conic_pdf, isfin

@dataclass(frozen=True)
class Config:
  tile_size: int = 16
  clamp_max_alpha: float = 0.99
  alpha_threshold: float = 1. / 255.
  saturate_threshold: float = 0.9999


@cache
def forward_kernel(config: Config, feature_size: int):

  feature_vec = ti.types.vector(feature_size, dtype=ti.f32)
  tile_size = config.tile_size

  @ti.kernel
  def _forward_kernel(
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, 6)
      point_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)
      
      # (TH, TW, 2) the start/end (0..K] index of ranges in the overlap_to_point array
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      # (K) ranges of points mapping to indexes into points list
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
      
      # outputs
      image_feature: ti.types.ndarray(feature_vec, ndim=2),  # (H, W, F)
      # needed for backward
      image_alpha: ti.types.ndarray(ti.f32, ndim=2),       # H, W
      image_last_valid: ti.types.ndarray(ti.i32, ndim=2),  # H, W
  ):

    camera_height, camera_width = image_feature.shape

    tile_area = ti.static(tile_size * tile_size)
    tiles_wide, tiles_high = camera_width // tile_size, camera_height // tile_size

    # put each tile_size * tile_size tile in the same CUDA thread group (block)
    ti.loop_config(block_dim=(tile_area))
    for tile_v, tile_u, v, u in ti.ndrange(tiles_high, tiles_wide, tile_size, tile_size):

      pixel = ivec2(tile_u, tile_v) * tile_size + ivec2(u, v) 
      tile_id = tile_u + tile_v * tiles_wide
      thread_id = u + v * tile_size

      # The initial value of accumulated alpha (initial value of accumulated multiplication)
      T_i = 1.0
      accum_feature = feature_vec(0.)

      # open the shared memory
      tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
      tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)

      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset

      num_point_groups = (tile_point_count + ti.static(tile_area - 1)) // tile_area
      pixel_saturated = False
      last_point_idx = start_offset

      # Loop through the range in groups of tile_area
      for point_group_id in range(num_point_groups):
        # The original implementation uses a predicate block the next update for shared memory until all threads finish the current update
        # but it is not supported by Taichi yet, and experiments show that it does not affect the performance
        tile_saturated = ti.simt.block.sync_all_nonzero(predicate=ti.cast(
            pixel_saturated, ti.i32))
        if tile_saturated != 0:
          continue

        ti.simt.block.sync()

        # The offset of the first point in the group
        group_start_offset = start_offset + point_group_id * tile_area

        # each thread in a block loads one point into shared memory
        # then all threads in the block process those points sequentially
        load_index = group_start_offset + thread_id
        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]

          tile_point[thread_id] = points[point_idx]
          tile_feature[thread_id] = point_features[point_idx]


        ti.simt.block.sync()

        max_point_group_offset: ti.i32 = ti.min(
            tile_area, tile_point_count - point_group_id * tile_area)

        # in parallel across a block, render all points in the group
        for in_group_idx in range(max_point_group_offset):
          if pixel_saturated or in_group_idx >= max_point_group_offset:
            break

          uv, uv_conic, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])
          gaussian_alpha = conic_pdf(ti.cast(pixel, ti.f32) + 0.5, uv, uv_conic)
          alpha = point_alpha * gaussian_alpha

            
          # from paper: we skip any blending updates with 𝛼 < 𝜖 (we choose 𝜖 as 1
          # 255 ) and also clamp 𝛼 with 0.99 from above.
          if alpha < ti.static(config.alpha_threshold):
            continue

          alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))
          # from paper: before a Gaussian is included in the forward rasterization
          # pass, we compute the accumulated opacity if we were to include it
          # and stop front-to-back blending before it can exceed 0.9999.
          next_T_i = T_i * (1 - alpha)
          if next_T_i < ti.static(1 - ti.static(config.saturate_threshold)):
            pixel_saturated = True
            continue  # somehow faster than directly breaking
          last_point_idx = group_start_offset + in_group_idx + 1

          # weight = alpha * T_i
          accum_feature += tile_feature[in_group_idx] * alpha * T_i
          T_i = next_T_i
        # end of point group loop
      # end of point group id loop

      image_feature[pixel.y, pixel.x] = accum_feature

      # No need to accumulate a normalisation factor as it is exactly 1 - T_i
      image_alpha[pixel.y, pixel.x] = 1. - T_i    
      image_last_valid[pixel.y, pixel.x] = last_point_idx

    # end of pixel loop

  return _forward_kernel




