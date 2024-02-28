import argparse
from functools import partial

import torch
from taichi_splatting.benchmarks.util import benchmarked
from taichi_splatting import spherical_harmonics

import taichi as ti



def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=200)
  parser.add_argument('--degree', type=int, default=3)
  
  
  args = parser.parse_args(args)
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args



def bench_sh(args):

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
        device_memory_GB=0.1)
  
  torch.manual_seed(args.seed)

  with torch.no_grad():
    sh_features = torch.randn(args.n, 3, (args.degree+1)**2, device=args.device).to(args.device)
    points = torch.randn(args.n, 3, device=args.device).to(args.device)

    indexes = torch.arange(args.n, device=args.device)
    
    camera_pos = torch.zeros(3, device=args.device)

    forward = partial(spherical_harmonics.evaluate_sh_at, sh_features, points, indexes, camera_pos)
    benchmarked('forward', forward, profile=args.profile, iters=args.iters)  

  def backward():
    colors = spherical_harmonics.evaluate_sh_at(sh_features, points, indexes, camera_pos)
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