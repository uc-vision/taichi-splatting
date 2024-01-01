
import taichi as ti
from taichi.lang.simt import block, warp

@ti.func
def interleave(x:ti.i32):
    x = ( x | ( x << 4 ) ) & 0x0F0F0F0F
    x = ( x | ( x << 2 ) ) & 0x33333333
    x = ( x | ( x << 1 ) ) & 0x55555555
    return x

@ti.func
def morton_tile(x:ti.i32, y:ti.i32, tile_size):
    order = interleave(x) | ( interleave(y) << 1 )
    return order // tile_size, order % tile_size


warp_size = 32

@ti.func
def warp_sum_f32(val: ti.f32):
  global_tid = block.global_thread_idx()
  mask_full = ti.u32(0xFFFFFFFF)

  lane_id = global_tid % warp_size
  offset_j = 16
  for j in ti.static(range(5)):
      n = warp.shfl_down_f32(mask_full, val, offset_j)
      if lane_id < offset_j:
          val += n
      offset_j = offset_j // 2
  
  return val

@ti.func
def warp_sum_vector(val: ti.template()):
  for i in ti.static(range(val.n)):
    val[i] = warp_sum_f32(val[i])
  
  global_tid = block.global_thread_idx()
  return (global_tid % warp_size) == 0

@ti.func
def atomic_add_vector(dest:ti.template(), val: ti.template()):
  # work around taichi bug for atomic_add with shared memory
  for i in ti.static(range(dest.n)):
    ti.atomic_add(dest[i], val[i])


@ti.func
def warp_add_vector(dest:ti.template(), val: ti.template()):
  if warp_sum_vector(val):
    # ti.atomic_add(dest, val) 
    # taichi bug makes this spit out a compiler error when using shared memory
    atomic_add_vector(dest, val)