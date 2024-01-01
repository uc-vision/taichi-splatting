
from functools import partial
import operator
import taichi as ti
from taichi.lang.simt import block, warp

@ti.func
def interleave(x:ti.i32):
    x = ( x | ( x << 4 ) ) & 0x0F0F0F0F
    x = ( x | ( x << 2 ) ) & 0x33333333
    x = ( x | ( x << 1 ) ) & 0x55555555
    return x


@ti.func
def deinterleave(x:ti.i32):
  x &= 0x55555555                   
  x = (x ^ (x >>  1)) & 0x33333333 
  x = (x ^ (x >>  2)) & 0x0f0f0f0f
  return x


@ti.func
def morton_tile(x:ti.i32, y:ti.i32):
    order = interleave(x) | ( interleave(y) << 1 )
    return order

@ti.func
def morton_tile_inv(order:ti.i32):
  x = deinterleave(order)
  y = deinterleave(order >> 1)
  return x, y



warp_size = 32

@ti.func
def warp_reduce_f32(val: ti.f32, op:ti.template()):
  global_tid = block.global_thread_idx()
  mask_full = ti.u32(0xFFFFFFFF)

  lane_id = global_tid % warp_size
  offset_j = 16
  for offset_j in ti.static([16, 8, 4, 2, 1]):
      n = warp.shfl_down_f32(mask_full, val, offset_j)
      if lane_id < offset_j:
          val = op(val, n)
  
  return val



@ti.func
def warp_reduce_vector(val: ti.template(), op:ti.template()):
  for i in ti.static(range(val.n)):
    val[i] = warp_reduce_f32(val[i], op)
  
  global_tid = block.global_thread_idx()
  return (global_tid % warp_size) == 0


@ti.func
def warp_reduce_i32(val: ti.i32, op:ti.template()):
  global_tid = block.global_thread_idx()
  mask_full = ti.u32(0xFFFFFFFF)

  lane_id = global_tid % warp_size
  offset_j = 16
  for offset_j in ti.static([16, 8, 4, 2, 1]):
      n = warp.shfl_down_i32(mask_full, val, offset_j)
      if lane_id < offset_j:
          val = op(val, n)
  
  return val

@ti.func
def is_warp_leader():
  global_tid = block.global_thread_idx()
  return (global_tid % warp_size) == 0

@ti.func 
def block_reduce_i32(val: ti.i32, op:ti.template(), atomic_op:ti.template(), initial:ti.i32):

  shared = ti.simt.block.SharedArray(1, dtype=ti.i32)
  shared[0] = ti.i32(initial)

  ti.simt.block.sync()
  warp_min = warp_reduce_i32(val, op)
  if is_warp_leader():
    atomic_op(shared[0], warp_min)

  ti.simt.block.sync()
  return shared[0]



@ti.func
def atomic_add_vector(dest:ti.template(), val: ti.template()):
  # work around taichi bug for atomic_add with shared memory
  for i in ti.static(range(dest.n)):
    ti.atomic_add(dest[i], val[i])

@ti.func
def add(a, b):
  return a + b



@ti.func
def warp_add_vector(dest:ti.template(), val: ti.template()):
  if warp_reduce_vector(val, add):
    # ti.atomic_add(dest, val) 
    # taichi spits out a LLVM assertion when using shared memory
    atomic_add_vector(dest, val)