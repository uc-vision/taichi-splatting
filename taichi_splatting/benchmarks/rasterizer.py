import argparse
from functools import partial

import torch
from tqdm import tqdm
from taichi_splatting.rasterizer.function import rasterize, RasterConfig

from taichi_splatting.renderer2d import project_gaussians2d
import taichi as ti
from taichi_splatting.scripts.fit_image_gaussians import random_2d_gaussians

import time

from taichi_splatting.tile_mapper import map_to_tiles


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  

  args = parser.parse_args()
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args

def benchmarked(name, f, iters=10, warmup=1, profile=False):

  for _ in range(warmup):
    f()

  if profile:
    ti.profiler.clear_kernel_profiler_info()

  start = time.time()
  for _ in tqdm(range(iters), desc=f"{name}"):
    f()

  torch.cuda.synchronize()
  ti.sync()

  end = time.time()

  print(f'{name}  {iters} iterations: {end - start:.3f}s at {iters / (end - start):.1f} iters/sec')

  if  profile:
    ti.profiler.print_kernel_profiler_info("count")


def main():

  args = parse_args()

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
        device_memory_GB=0.1, kernel_profiler=True)

  conditions = [(scale_factor, n, tile_size)
                for scale_factor in [1, 2, 4]
                for n in [500000, 1000000]
                for tile_size in [16, 32]]
  
     
  for i, (scale_factor, n, tile_size) in enumerate(conditions):
    torch.manual_seed(i)

    gaussians = random_2d_gaussians(n, args.image_size, 
            scale_factor, alpha_range=(0.5, 1.0)).to(args.device)
    
    
    gaussians2d = project_gaussians2d(gaussians)
    tile_map = partial(map_to_tiles, gaussians2d, gaussians.depth, 
      image_size=args.image_size, tile_size=tile_size)
    
    overlap_to_point, ranges = tile_map()

    points_per_tile = (ranges[1] - ranges[0]).float().mean()
    point_overlap = overlap_to_point.shape[0] / n 
    

    print('----------------------------------------------------------')
    print(f'scale_factor={scale_factor}, n={n}, tile_size={tile_size} point_overlap={point_overlap:.2f} tile_points={points_per_tile:.2f}')
    

    benchmarked('map_to_tiles', tile_map, profile=args.profile)  

    raster_config = RasterConfig(tile_size=tile_size)

    forward = partial(rasterize, gaussians=gaussians2d, features=gaussians.feature, 
      tile_overlap_ranges=ranges, overlap_to_point=overlap_to_point,
      image_size=args.image_size, config=raster_config)
    
    benchmarked('forward', forward, profile=args.profile)  

    gaussians2d.requires_grad_(True)
    def backward():
      image, alpha = forward()
      image.sum().backward()

    benchmarked('backward', backward, profile=args.profile)  

if __name__ == '__main__':
  main()