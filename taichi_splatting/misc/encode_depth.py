from beartype.typing import Tuple
import taichi as ti
import torch

@ti.func
def encode_depth16_func(depth: ti.f32, near:ti.f32, far:ti.f32) -> ti.i16:
  ndc_depth =  (far + near - (2.0 * near * far) / depth) / (far - near)
  return ti.cast(ti.math.clamp(ndc_depth, -1.0, 1.0) * 32767, ti.i16)  


@ti.kernel
def encode_depth16_kernel(
  depths:ti.types.ndarray(ti.f32, ndim=2), near:ti.f32, far:ti.f32, 
  encoded:ti.types.ndarray(ti.i16, ndim=1)):

  for i in range(encoded.shape[0]):
    encoded[i] = encode_depth16_func(depths[i, 0], near, far)


def encode_depth16(depths:torch.Tensor, depth_range:Tuple[float, float] ) -> torch.Tensor:
  encoded = torch.empty((depths.shape[0], ), dtype=torch.int16, device=depths.device)
  
  encode_depth16_kernel(depths, depth_range[0], depth_range[1], encoded)
  return encoded


def encode_depth32(depths:torch.Tensor) -> torch.Tensor:
  return depths[:, 0].view(dtype=torch.int32)
  
def encode_depth(depths:torch.Tensor, depth_range:Tuple[float, float], use_depth16=False ) -> torch.Tensor:
  if use_depth16:
    return encode_depth16(depths, depth_range)
  else:
    return encode_depth32(depths)