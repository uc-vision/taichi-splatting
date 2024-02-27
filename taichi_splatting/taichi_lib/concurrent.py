

import taichi as ti
from taichi.lang.simt import block, warp



WARP_SIZE = 32

@ti.func
def warp_reduce_f32(val: ti.f32, op:ti.template()):
  global_tid = block.global_thread_idx()
  mask_full = ti.u32(0xFFFFFFFF)

  lane_id = global_tid % WARP_SIZE
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
  return (global_tid % WARP_SIZE) == 0


@ti.func
def warp_reduce_i32(val: ti.i32, op:ti.template()):
  global_tid = block.global_thread_idx()
  mask_full = ti.u32(0xFFFFFFFF)

  lane_id = global_tid % WARP_SIZE
  for offset_j in ti.static([16, 8, 4, 2, 1]):
      n = warp.shfl_down_i32(mask_full, val, offset_j)
      if lane_id < offset_j:
          val = op(val, n)
  
  return val

@ti.func
def is_warp_leader():
  global_tid = block.global_thread_idx()
  return (global_tid % WARP_SIZE) == 0

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
def warp_add_vector_32(dest:ti.template(), val: ti.template()):
  if warp_reduce_vector(val, add):
    # ti.atomic_add(dest, val) 
    # taichi spits out a LLVM assertion when using shared memory
    atomic_add_vector(dest, val)


@ti.func
def warp_add_vector_64(dest:ti.template(), val: ti.template()):
  # placeholder for testing 64 bit functions only
  return atomic_add_vector(dest, val)


@ti.func
def warp_scan_up(val: ti.template(), op:ti.template()):
    global_tid = block.global_thread_idx()
    lane_id = global_tid % WARP_SIZE
    mask_full = ti.u32(0xFFFFFFFF)

    out = val
    for offset_j in ti.static([16]):#, 8, 4, 2, 1]):
      n = warp.shfl_up_i32(mask_full, out, offset_j)
      if lane_id >= offset_j:
          out = op(out, n)

    return out


@ti.func
def warp_scan_down(val: ti.template(), op:ti.template()):
    global_tid = block.global_thread_idx()
    lane_id = global_tid % WARP_SIZE
    mask_full = ti.u32(0xFFFFFFFF)

    out = val
    for offset_j in ti.static([16, 8, 4, 2, 1]):
      n = warp.shfl_down_i32(mask_full, out, offset_j)
      if 31 - lane_id >= offset_j:
        out = op(out, n)

    return out


@ti.func
def warp_allocate(dest:ti.template(), val: ti.template()):
  mask_full = ti.u32(0xFFFFFFFF)
  global_tid = block.global_thread_idx()
  warp_prefix = warp_scan_down(val, add)

  prev = 0
  if (global_tid % WARP_SIZE) == 0:
    prev = ti.atomic_add(dest, warp_prefix)
    warp.shfl_sync_i32(mask_full, prev, 0)
  else:
    prev = warp.shfl_sync_i32(mask_full, 0, 0)

  return prev + (warp_prefix - val)

