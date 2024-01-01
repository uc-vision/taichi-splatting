import argparse
from functools import partial

import torch
from taichi_splatting.benchmarks.util import benchmarked
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting import projection
# from taichi_splatting import projection


import taichi as ti
from taichi_splatting.tests.random_data import random_3d_gaussians, random_camera



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=1000)
  

  args = parser.parse_args()
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args


def main():

  args = parse_args()

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
        device_memory_GB=0.1, kernel_profiler=args.profile)
  
     
  torch.manual_seed(args.seed)

  with torch.no_grad():
    camera_params = random_camera()
    gaussians = random_3d_gaussians(args.n, camera_params)

    gaussians, camera_params = gaussians.to(args.device), camera_params.to(args.device)
    packed = gaussians.packed()

    project_forward = partial(projection.apply, packed, camera_params.T_image_camera, camera_params.T_camera_world)
    benchmarked('forward', project_forward, profile=args.profile, iters=args.iters)  


  packed.requires_grad_(True)

  def project_backward():
    points, depth = projection.apply(packed, camera_params.T_image_camera, camera_params.T_camera_world)
    loss = points.sum() + depth.sum()
    loss.backward()

  
  benchmarked('backward (gaussians)', project_backward, profile=args.profile, iters=args.iters)  


  packed.requires_grad_(False)    
  camera_params.T_camera_world.requires_grad_(True)
  camera_params.T_image_camera.requires_grad_(True)
  benchmarked('backward (camera)', project_backward, profile=args.profile, iters=args.iters)  



if __name__ == '__main__':
  main()
