import argparse
from functools import partial

import torch
from taichi_splatting.benchmarks.util import benchmarked
from taichi_splatting import indexed_spherical_harmonics

import gstaichi as ti

from taichi_splatting.taichi_queue import TaichiQueue, taichi_queue



def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=200)
  parser.add_argument('--degree', type=int, default=3)
  parser.add_argument('--debug', action='store_true')
  
  args = parser.parse_args(args)
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args



def bench_sh(args):
  with taichi_queue(arch=ti.cuda, log_level=ti.INFO if not args.debug else ti.DEBUG, debug=args.debug):
    torch.manual_seed(args.seed)

    with torch.no_grad():
      sh_features = torch.randn(args.n, 3, (args.degree+1)**2, device=args.device).to(args.device)
      points = torch.randn(args.n, 3, device=args.device).to(args.device)

      indexes = torch.arange(args.n, device=args.device)
      
      camera_pos = torch.zeros(3, device=args.device)

      forward = partial(indexed_spherical_harmonics.evaluate_sh_at, sh_features, points, indexes, camera_pos)
      benchmarked('forward', forward, profile=args.profile, iters=args.iters)  

    def backward():
      colors = indexed_spherical_harmonics.evaluate_sh_at(sh_features, points, indexes, camera_pos)
      loss = colors.sum()
      loss.backward()

    sh_features.requires_grad_(True)
    benchmarked('backward (sh_features)', backward, profile=args.profile, iters=args.iters)  

    sh_features.requires_grad_(True)
    points.requires_grad_(True)
    camera_pos.requires_grad_(True)
    benchmarked('backward (all)', backward, profile=args.profile, iters=args.iters)  


def main():
  args = parse_args()
  bench_sh(args)

if __name__ == '__main__':
  main()