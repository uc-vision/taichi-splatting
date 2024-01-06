import argparse
from functools import partial

import torch
import taichi as ti
from taichi_splatting.benchmarks.util import benchmarked

from taichi_splatting.rasterizer.function import rasterize_with_tiles, RasterConfig
from taichi_splatting.renderer2d import project_gaussians2d
from taichi_splatting.scripts.fit_image_gaussians import random_2d_gaussians
from taichi_splatting.tile_mapper import map_to_tiles


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--scale_factor', type=int, default=2)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=100)

  

  args = parser.parse_args()
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args


def test_rasterizer():

  args = parse_args()

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
        device_memory_GB=0.1)
  
     
  torch.manual_seed(args.seed)

  gaussians = random_2d_gaussians(args.n, args.image_size, 
          args.scale_factor, alpha_range=(0.5, 1.0)).to(args.device)
  config = RasterConfig(tile_size=args.tile_size)

  
  gaussians2d = project_gaussians2d(gaussians)
  tile_map = partial(map_to_tiles, gaussians2d, gaussians.depth, 
    image_size=args.image_size, 
    config=config)
  
  overlap_to_point, ranges = tile_map()

  points_per_tile = (ranges[1] - ranges[0]).float().mean()
  point_overlap = overlap_to_point.shape[0] / args.n 
  

  print('----------------------------------------------------------')
  print(f'scale_factor={args.scale_factor}, n={args.n}, tile_size={args.tile_size} point_overlap={point_overlap:.2f} tile_points={points_per_tile:.2f}')
  
  benchmarked('map_to_tiles', tile_map, profile=args.profile, iters=args.iters)  


  forward = partial(rasterize_with_tiles, gaussians2d=gaussians2d, features=gaussians.feature, 
    tile_overlap_ranges=ranges, overlap_to_point=overlap_to_point,
    image_size=args.image_size, config=config)
  
  benchmarked('forward', forward, profile=args.profile, iters=args.iters)  

  gaussians.feature.requires_grad_(True)
  def backward():
    image, alpha = forward()
    image.sum().backward()

  benchmarked('backward (features)', backward, profile=args.profile, iters=args.iters)  

  gaussians.feature.requires_grad_(False)
  gaussians2d.requires_grad_(True)

  benchmarked('backward (gaussians)', backward, profile=args.profile, iters=args.iters)  

  gaussians.feature.requires_grad_(True)
  gaussians2d.requires_grad_(True)

  benchmarked('backward (all)', backward, profile=args.profile, iters=args.iters)  


if __name__ == '__main__':
  main()