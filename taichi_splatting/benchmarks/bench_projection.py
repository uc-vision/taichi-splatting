import argparse
from functools import partial

import torch
from taichi_splatting.benchmarks.util import benchmarked
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.perspective import projection
# from taichi_splatting import projection


import taichi as ti
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_3d_gaussians, random_camera


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=500)
  parser.add_argument('--margin', type=float, default=0.0, help="controls random points (non visible) margin")
  

  args = parser.parse_args(args)
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args


def bench_projection(args):

  TaichiQueue.init(arch=ti.cuda, offline_cache=True, log_level=ti.INFO, 
        device_memory_GB=0.1)
  
     
  torch.manual_seed(args.seed)

  with torch.no_grad():
    camera_params = random_camera()
    gaussians = random_3d_gaussians(args.n, camera_params, margin=args.margin)
    config = RasterConfig()

    gaussians, camera_params = gaussians.to(args.device), camera_params.to(args.device)


    _, _, vis_idx = projection.project_to_image(gaussians, camera_params, config)
    print(args)
    print(f"benchmarking {args.n} points ({vis_idx.shape[0]} visible) points")



    project_forward = partial(projection.project_to_image, gaussians, camera_params, config)
    benchmarked('forward', project_forward, profile=args.profile, iters=args.iters)  


  gaussians.requires_grad_(True)

  def project_backward():
    points, depth, indexes = projection.project_to_image(gaussians, camera_params, config)
    loss = points.sum() + depth.sum()
    loss.backward()

  
  benchmarked('backward (gaussians)', project_backward, profile=args.profile, iters=args.iters)  


  gaussians.requires_grad_(False)
  camera_params.T_camera_world.requires_grad_(True)
  benchmarked('backward (extrinsics)', project_backward, profile=args.profile, iters=args.iters)  

  camera_params.T_camera_world.requires_grad_(False)
  camera_params.projection.requires_grad_(True)
  benchmarked('backward (intrinsics)', project_backward, profile=args.profile, iters=args.iters)  

  gaussians.requires_grad_(True)
  camera_params.T_camera_world.requires_grad_(True)
  camera_params.projection.requires_grad_(True)
  benchmarked('backward (everything)', project_backward, profile=args.profile, iters=args.iters)  


def main():
  args = parse_args()
  bench_projection(args)

if __name__ == '__main__':
  main()
