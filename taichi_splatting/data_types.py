from dataclasses import dataclass
from numbers import Integral
from typing import Tuple
from beartype import beartype
from tensordict import tensorclass
import torch

  

@dataclass(frozen=True)
class RasterConfig:
  tile_size: int = 16
  margin_tiles: int = 3

  # cutoff N standard deviations from mean
  gaussian_scale: float = 3.0   
  
  # cull to an oriented box, otherwise an axis aligned bounding box
  tight_culling: bool = True  

  clamp_max_alpha: float = 0.99
  alpha_threshold: float = 1. / 255.
  saturate_threshold: float = 0.9999


def check_packed3d(packed_gaussians: torch.Tensor):
  assert len(packed_gaussians.shape) == 2 and packed_gaussians.shape[1] == 11, f"Expected shape (N, 11), got {packed_gaussians.shape}"  

def check_packed2d(packed_gaussians: torch.Tensor):
  assert len(packed_gaussians.shape) == 2 and packed_gaussians.shape[1] == 6, f"Expected shape (N, 6), got {packed_gaussians.shape}"  



@tensorclass
class Gaussians3D():
  position     : torch.Tensor # 3  - xyz
  log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 4  - quaternion wxyz
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  feature      : torch.Tensor # (any rgb (3), spherical harmonics (3x16) etc)


  def __post_init__(self):
    assert self.position.shape[1] == 3, f"Expected shape (N, 3), got {self.position.shape}"
    assert self.log_scaling.shape[1] == 3, f"Expected shape (N, 3), got {self.log_scaling.shape}"
    assert self.rotation.shape[1] == 4, f"Expected shape (N, 4), got {self.rotation.shape}"
    assert self.alpha_logit.shape[1] == 1, f"Expected shape (N, 1), got {self.alpha_logit.shape}"


  def packed(self):
    return torch.cat([self.position, self.log_scaling, self.rotation, self.alpha_logit], dim=-1)

  @property
  def scale(self):
    return torch.exp(self.log_scaling)

  @property
  def alpha(self):
    return torch.sigmoid(self.alpha_logit)

@tensorclass
class Gaussians2D():
  position     : torch.Tensor # 2  - xy
  depth        : torch.Tensor # 1  - for sorting
  log_scaling   : torch.Tensor # 2  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 2  - unit length imaginary number
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  feature      : torch.Tensor # N  - (any rgb, label etc)

  @staticmethod
  def unpack_vec(vec):
    uv, uv_conic, alpha = vec[:, 0:2], vec[:, 2:5], vec[:, 5]
    return uv, uv_conic, alpha


@beartype
@dataclass
class CameraParams:
  T_image_camera: torch.Tensor # (3, 3) camera projection matrix
  T_camera_world  : torch.Tensor # (4, 4) camera view matrix

  @property
  def device(self):
    return self.T_image_camera.device

  @property
  def T_image_world(self):
    T_image_camera = torch.eye(4, 
      device=self.T_image_camera.device, dtype=self.T_image_camera.dtype)
    T_image_camera[0:3, 0:3] = self.T_image_camera

    return T_image_camera @ self.T_camera_world

  near_plane: float
  far_plane: float
  image_size: Tuple[Integral, Integral]
  orthographic: bool = False

  def __repr__(self):
    w, h = self.image_size
    fx, fy = self.T_image_camera[0, 0], self.T_image_camera[1, 1]
    cx, cy = self.T_image_camera[0, 2], self.T_image_camera[1, 2]

    pos_str = ", ".join([f"{x:.3f}" for x in self.camera_position])
    return f"CameraParams({w}x{h}, fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}, clipping={self.near_plane:.4f}-{self.far_plane:.4f}, position=({pos_str})"
  

  @property
  def camera_position(self):
    T_world_camera = torch.inverse(self.T_camera_world)
    return T_world_camera[0:3, 3]

  def to(self, device=None, dtype=None):
    return CameraParams(
      T_image_camera=self.T_image_camera.to(device=device, dtype=dtype),
      T_camera_world=self.T_camera_world.to(device=device, dtype=dtype),
      near_plane=self.near_plane,
      far_plane=self.far_plane,
      image_size=self.image_size
    )

  def __post_init__(self):
    assert self.T_image_camera.shape == (3, 3), f"Expected shape (3, 3), got {self.T_image_camera.shape}"
    assert self.T_camera_world.shape == (4, 4), f"Expected shape (4, 4), got {self.T_camera_world.shape}"

    assert len(self.image_size) == 2
    assert self.near_plane > 0
    assert self.far_plane > self.near_plane



