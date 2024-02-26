import taichi as ti

from  . import f32
from  . import f64



def get_library(dtype):
  if dtype == ti.f32:
    return f32
  elif dtype == ti.f64:
    return f64
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")
  
  