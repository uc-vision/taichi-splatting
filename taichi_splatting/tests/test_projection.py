import math

from taichi_splatting.data_types import CameraParams, Gaussians3D
from taichi_splatting.tests.util import compare_with_grad
import taichi_splatting.torch_ops.projection as torch_proj
from taichi_splatting.torch_ops.util import check_finite

import taichi_splatting.projection as ti_proj

import torch

from taichi_splatting.torch_ops.transforms import join_rt
import taichi as ti

ti.init(debug=True)


def random_camera(pos_scale:float=100., max_image_size:int = 1024) -> CameraParams:
  q = torch.randn((1, 4))
  t = torch.randn((3)) * pos_scale

  T_world_camera = join_rt(torch_proj.quat_to_mat(q), t)
  T_camera_world = torch.inverse(T_world_camera)

  w, h = [x.item() for x in torch.randint(size=(2,), 
            low=max_image_size // 5, high=max_image_size)]
  cx, cy = torch.tensor([w/2, h/2]) + torch.randn(2) * (w / 5) 

  fov = torch.deg2rad(torch.rand(1) * 70 + 30)
  fx = w / (2 * torch.tan(fov / 2))
  fy = h / (2 * torch.tan(fov / 2))

  T_image_camera = torch.tensor([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
  ])

  near_plane = math.exp(10 * -torch.rand(1).item()) # 1e-5 to 1.0
  assert near_plane > 0

  return CameraParams(
    T_camera_world=T_camera_world,
    T_image_camera=T_image_camera,
    image_size=(w, h),
    near_plane=near_plane,
    far_plane=near_plane * 1000.
  )


def random_3d_gaussians(n, camera_params:CameraParams) -> Gaussians3D:
  w, h = camera_params.image_size

  uv_pos = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)

  depth_range = camera_params.far_plane - camera_params.near_plane
  depth = torch.rand(n) * depth_range + camera_params.near_plane   

  print(depth_range, camera_params.far_plane, camera_params.near_plane)

  position = torch_proj.unproject_points(uv_pos, depth.unsqueeze(1), camera_params.T_image_world)
  fx = camera_params.T_image_camera[0, 0]

  scale =  (w / math.sqrt(n)) * (depth / fx) 
  scaling = (torch.rand(n, 3) + 0.2) * scale.unsqueeze(1) 

  rotation = torch.randn(n, 4) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

  alpha = torch.rand(n) * 0.8 + 0.1

  return Gaussians3D(
    position=position,
    log_scaling=torch.log(scaling),
    rotation=rotation,
    alpha_logit=torch_proj.inverse_sigmoid(alpha).unsqueeze(1),
    feature=torch.rand(n, 3),
    batch_size=(n,)
  )


def random_inputs(device='cpu', max_points=10):
  def f(seed:int = 0):
    torch.manual_seed(seed)
    camera = random_camera()
    n = torch.randint(size=(1,), low=1, high=max_points).item()


    gaussians = random_3d_gaussians(n=n, camera_params=camera)
    check_finite(gaussians)


    return (gaussians.pack_gaussian3d().to(device), 
      camera.T_image_camera.to(device), camera.T_camera_world.to(device))
  return f


def test_projection(iters = 100):
  gen_inputs = random_inputs()

  compare_with_grad("projection", 
    input_names=["gaussians", "T_image_camera", "T_camera_world"],
    output_names=["points", "depths"],
    f1=torch_proj.apply, 
    f2=ti_proj.apply, 
    gen_inputs=gen_inputs, iters=iters)



if __name__ == '__main__':
  torch.set_printoptions(precision=4, sci_mode=False)
  test_projection()