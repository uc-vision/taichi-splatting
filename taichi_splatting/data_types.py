from dataclasses import dataclass, replace
import math
from typing import List
from beartype.typing import Tuple
from beartype import beartype
import roma
from taichi_splatting.torch_lib.transforms import split_rt, transform44, make_homog
import torch


from torch.nn import functional as F
from tensordict import TensorClass


  
@beartype
@dataclass(frozen=True, eq=True, kw_only=True)
class RasterConfig:
  tile_size: int = 16

  # pixel tilin per thread in the backwards pass 
  pixel_stride: Tuple[int, int] = (2, 2)


  # clamp position to within this margin of the image for affine jaocbian
  clamp_margin: float = 0.15  
  
  # Use anti-aliasing implementation
  antialias: bool = False  

  # blur covariance matrix by this factor
  blur_cov: float = 0.3

  clamp_max_alpha: float = 0.99
  alpha_threshold: float = 1. / 255.

  # stop alpha blending at this point
  saturate_threshold: float = 0.9999

  # use alpha blending - if set to false, with saturate_threshold can be used to compute quantile (e.g. median)
  use_alpha_blending: bool = True

  compute_point_heuristic: bool = False # implies compute_visibility
  compute_visibility: bool = False # compute visibility (pixels) for each gaussian

  median_threshold: float = 0.25 # threshold for median depth computation

    

def check_packed3d(packed_gaussians: torch.Tensor):
  assert len(packed_gaussians.shape) == 2 and packed_gaussians.shape[1] == 11, f"Expected shape (N, 11), got {packed_gaussians.shape}"  

def check_packed2d(packed_gaussians: torch.Tensor):
  assert len(packed_gaussians.shape) == 2 and packed_gaussians.shape[1] == 6, f"Expected shape (N, 6), got {packed_gaussians.shape}"  


class Gaussians3D(TensorClass):
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
  
  def shape_tensors(self):
    return (self.position, self.log_scaling, self.rotation, self.alpha_logit)

  def scaled(self, scale:float) -> 'Gaussians3D':
    return self.replace(
      position=self.position * scale,
      log_scaling=math.log(scale) + self.log_scaling)

  def translated(self, translation:torch.Tensor) -> 'Gaussians3D':
    return self.replace(position=self.position + translation.view(1, 3))

  @property
  def scale(self):
    return torch.exp(self.log_scaling)
  

  def transform_rigid(self, m:torch.Tensor):
    """ Transform the gaussians by a 4x4 matrix """
    assert m.shape == (4, 4), f"Expected shape (4, 4), got {m.shape}"

    position = transform44(m, make_homog(self.position))[..., 0:3]
    
    r, _ = split_rt(m)  
    rotation = r @ roma.unitquat_to_rotmat(self.rotation)

  
    return self.replace(position=position, 
                        rotation=roma.rotmat_to_unitquat(rotation))

  @property
  def alpha(self):
    return torch.sigmoid(self.alpha_logit)
  
  @staticmethod
  def concat_batch(gaussians:List['Gaussians3D']) -> 'Gaussians3D':
    dicts = [g.to_dict() for g in gaussians]
    dicts = {k: torch.cat([d[k] for d in dicts], dim=0) for k in dicts[0].keys()}

    batch_size = sum([g.batch_size[0] for g in gaussians])
    return Gaussians3D(**dicts, batch_size=(batch_size,))



def inverse_sigmoid(x:torch.Tensor):
  return torch.log(x / (1 - x))


class Gaussians2D(TensorClass):
  position     : torch.Tensor # 2  - xy
  depths        : torch.Tensor # 1  - for sorting
  log_scaling   : torch.Tensor # 2

  rotation      : torch.Tensor # 2  - unit length imaginary number
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  feature      : torch.Tensor # N  - (any rgb, label etc)

  @property
  def opacity(self):
    return self.alpha_logit.sigmoid()
  
  @property
  def scaling(self):
    return torch.exp(self.log_scaling)


  def set_scaling(self, scaling) -> 'Gaussians2D':
    return self.replace(log_scaling=torch.log(scaling))



