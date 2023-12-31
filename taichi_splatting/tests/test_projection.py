from tqdm import tqdm

from taichi_splatting.tests.util import eval_with_grad
import taichi_splatting.torch_ops.projection as torch_proj
from taichi_splatting.torch_ops.util import check_finite
import taichi_splatting.projection as ti_proj
from .random_data import random_camera, random_3d_gaussians

import torch
import taichi as ti

from torch.autograd.gradcheck import GradcheckError


ti.init(arch=ti.cpu, offline_cache=True, log_level=ti.INFO)


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

  test_projection_grad()
  test_projection()
