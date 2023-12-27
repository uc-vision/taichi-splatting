from taichi_splatting.taichi_lib.generic import make_library
import taichi as ti


funcs = make_library(ti.f32)

__all__ = list(funcs.__dict__.keys())
globals().update(funcs.__dict__)