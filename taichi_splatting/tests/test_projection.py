import math

from tqdm import tqdm

from taichi_splatting.data_types import CameraParams, Gaussians3D
from taichi_splatting.tests.util import eval_with_grad
import taichi_splatting.torch_ops.projection as torch_proj
from taichi_splatting.torch_ops.util import check_finite

import taichi_splatting.projection as ti_proj

import torch
from torch.autograd.gradcheck import GradcheckError

from taichi_splatting.torch_ops.transforms import join_rt
import taichi as ti

ti.init(arch=ti.cpu, offline_cache=True, log_level=ti.INFO)


def random_camera(pos_scale:float=1., max_image_size:int = 1024) -> CameraParams:
  q = torch.randn((1, 4))
  t = torch.randn((3)) * pos_scale

  T_world_camera = join_rt(torch_proj.quat_to_mat(q), t)
  T_camera_world = torch.inverse(T_world_camera)

  w, h = [x.item() for x in torch.randint(size=(2,), 
            low=max_image_size // 5, high=max_image_size)]
  cx, cy = torch.tensor([w/2, h/2]) + torch.randn(2) * (w / 20) 

  fov = torch.deg2rad(torch.rand(1) * 70 + 30)
  fx = w / (2 * torch.tan(fov / 2))
  fy = h / (2 * torch.tan(fov / 2))

  T_image_camera = torch.tensor([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
  ])

  near_plane = 0.1
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


def random_inputs(device='cpu', max_points=1000, dtype=torch.float32):
  def f(seed:int = 0):
    torch.manual_seed(seed)
    camera = random_camera()
    n = torch.randint(size=(1,), low=1, high=max_points).item()

    gaussians = random_3d_gaussians(n=n, camera_params=camera)
    check_finite(gaussians)

    return (x.to(device=device, dtype=dtype) for x in [gaussians, camera])
  return f


def compare(name, x, y, **kwargs):
  if not torch.allclose(x, y, **kwargs):
    print(f"x={x}")
    print(f"y={y}")

    raise AssertionError(f"{name} mismatch")

def compare_outputs(out1, out2):
  points1, depth1 = out1
  points2, depth2 = out2

  compare("uv",  points1[:, 0:2], points2[:, 0:2])
  compare("conic",  points1[:, 2:5], points2[:, 2:5])
  compare("alpha",  points1[:, 5], points2[:, 5])
  compare("depth",  depth1, depth2)

def compare_grads(grad1, grad2):
  
  gaussians1, image_camera1, camera_world1 = grad1
  gaussians2, image_camera2, camera_world2 = grad2

  compare("gaussians", gaussians1, gaussians2)
  compare("image_camera", image_camera1, image_camera2)
  compare("camera_world", camera_world1, camera_world2)

def test_projection(iters = 100, dtype=torch.float64):
  gen_inputs = random_inputs(max_points=1000, dtype=dtype)

  for i in tqdm(range(iters)):
    gaussians, camera = gen_inputs(i)
    inputs = (gaussians.pack_gaussian3d(), 
      camera.T_image_camera, camera.T_camera_world)

    out1, grad1 = eval_with_grad(ti_proj.apply, *inputs)
    out2, grad2 = eval_with_grad(torch_proj.apply, *inputs)

    try:
      compare_outputs(out1, out2)
      compare_grads(grad1, grad2)

    except AssertionError as e:
      print(f"seed={i}")
      print(f"camera={camera}")
      raise e

def test_projection_grad(iters = 100):
  gen_inputs = random_inputs(max_points=10, dtype=torch.float64)

  for i in tqdm(range(iters), desc="projection_gradcheck"):
      gaussians, camera = gen_inputs(i)
      inputs = [x.requires_grad_(True) for x in (gaussians.pack_gaussian3d(), 
        camera.T_image_camera, camera.T_camera_world)]
      
      try:
        torch.autograd.gradcheck(ti_proj.apply, inputs)
      except GradcheckError as e:
        print(f"seed={i}")
        print(f"camera={camera}")
        raise e

if __name__ == '__main__':
  torch.set_printoptions(precision=8, sci_mode=False)
  test_projection()
  test_projection_grad()
