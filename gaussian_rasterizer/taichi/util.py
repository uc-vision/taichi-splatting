from math import prod
import taichi as ti

def struct_size(ti_struct:ti.lang.struct.StructType):
  size = 0
  for k, v in ti_struct.members.items():
    if isinstance(v, ti.lang.matrix.VectorType):

      size += prod(v.get_shape())
    elif isinstance(v, ti.lang.struct.StructType):
      size += struct_size(v)
    else:
      size += 1
  return int(size)

@ti.func
def sigmoid(x:ti.f32):
    return 1. / (1. + ti.exp(-x))