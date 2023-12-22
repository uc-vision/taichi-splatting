from dataclasses import dataclass
from numbers import Integral
from typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from tensordict import tensorclass

from taichi.math import vec2, vec3, vec4

from taichi_splatting.ti.util import sigmoid, struct_size

@tensorclass
class Gaussians():
  position     : torch.Tensor # 3  - xyz
  log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 4  - quaternion wxyz
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  feature      : torch.Tensor # N  - (any rgb, spherical harmonics etc)


  def concat(self):
    return torch.cat([self.position, self.feature, self.log_scaling, self.rotation, self.alpha_logit], dim=-1)

@beartype
@dataclass
class CameraParams:
  T_image_camera: torch.Tensor # (1, 3, 3) camera projection matrix
  T_camera_world  : torch.Tensor # (1, 4, 4) camera view matrix

  @property
  def T_image_world(self):
    T_image_camera = torch.eye(4, device=self.T_image_camera.device, dtype=torch.float32)
    T_image_camera[0:3, 0:3] = self.T_image_camera

    return T_image_camera @ self.T_camera_world


  near_plane: float
  far_plane: float
  image_size: Tuple[Integral, Integral]

  def __post_init__(self):
    assert self.T_image_camera.shape == (1, 3, 3), f"Expected shape (1, 3, 3), got {self.T_image_camera.shape}"
    assert self.T_camera_world.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {self.T_camera_world.shape}"

    assert len(self.image_size) == 2
    assert self.near_plane > 0
    assert self.far_plane > self.near_plane



@ti.dataclass
class Gaussian2D:
    uv        : vec2
    uv_conic  : vec3
    alpha   : ti.f32



@ti.dataclass
class Gaussian3D:
    position   : vec3
    log_scaling : vec3
    rotation    : vec4
    alpha_logit : ti.f32

    @ti.func
    def alpha(self):
      return sigmoid(self.alpha_logit)

    @ti.func
    def scale(self):
        return ti.math.exp(self.log_scaling)


vec_g2d = ti.types.vector(struct_size(Gaussian2D), dtype=ti.f32)
vec_g3d = ti.types.vector(struct_size(Gaussian3D), dtype=ti.f32)


@ti.func
def to_vec_g2d(uv:vec2, uv_conic:vec3, alpha:ti.f32) -> vec_g2d:
  return vec_g2d(*uv, *uv_conic, alpha)

@ti.func
def to_vec_g3d(position:vec3, log_scaling:vec3, rotation:vec4, alpha_logit:ti.f32) -> vec_g3d:
  return vec_g3d(*position, *log_scaling, *rotation, alpha_logit)


@ti.func
def unpack_vec_g3d(vec:vec_g3d) -> Gaussian3D:
  return vec[0:3], vec[3:6], vec[6:10], vec[10]

@ti.func
def unpack_vec_g2d(vec:vec_g2d) -> Gaussian2D:
  return vec[0:2], vec[2:5], vec[5]



@ti.func
def from_vec_g3d(vec:vec_g3d) -> Gaussian3D:
  return Gaussian3D(vec[0:3], vec[3:6], vec[6:10], vec[10])

@ti.func
def from_vec_g2d(vec:vec_g2d) -> Gaussian2D:
  return Gaussian2D(vec[0:2], vec[2:5], vec[5])

def unpack_g2d_torch(vec:torch.Tensor):
  uv, uv_conic, alpha = vec[:, 0:2], vec[:, 2:5], vec[:, 5]
  return uv, uv_conic, alpha

# Taichi structs don't have static methods, but they can be added afterward
Gaussian2D.vec = vec_g2d
Gaussian2D.to_vec = to_vec_g2d
Gaussian2D.from_vec = from_vec_g2d
Gaussian2D.unpack = unpack_vec_g2d


Gaussian3D.vec = vec_g3d
Gaussian3D.to_vec = to_vec_g3d
Gaussian3D.from_vec = from_vec_g3d
Gaussian3D.unpack = unpack_vec_g3d

