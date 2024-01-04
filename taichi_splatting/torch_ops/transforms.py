from typing import Tuple
import torch

def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
  """ Convert quaternion to rotation matrix
  """
  x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  x2, y2, z2 = x*x, y*y, z*z

  return torch.stack([
    1 - 2*y2 - 2*z2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
    2*x*y + 2*w*z, 1 - 2*x2 - 2*z2, 2*y*z - 2*w*x,
    2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x2 - 2*y2
  ], dim=-1).reshape(quat.shape[:-1] + (3, 3))

def split_rt(
    transform: torch.Tensor,  # (batch_size, 4, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    return R.contiguous(), t.contiguous()

def join_rt(r, t):
  T = torch.eye(4, device=r.device, dtype=r.dtype)
  T[0:3, 0:3] = r
  T[0:3, 3] = t
  return T
  

def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=points.dtype, device=points.device)], axis=-1)

def transform44(transform, points):

  points = points.reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ points
  return transformed[..., 0].reshape(-1, 4)



def transform33(transform, points):

  points = points.reshape([-1, 3, 1])
  transformed = transform.reshape([1, 3, 3]) @ points
  return transformed[..., 0].reshape(-1, 3)


