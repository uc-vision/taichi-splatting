import cv2
import argparse
import taichi as ti

import torch
from torch.optim import Adam
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.renderer2d import render_gaussians, Gaussians2D
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_ops.util import check_finite

import time


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--n', type=int, default=20000)
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')
  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--epoch', type=int, default=100, help='Number of iterations per measurement/profiling')
  

  return parser.parse_args()


def optimizer(gaussians: Gaussians2D, base_lr=1.0):

  learning_rates = dict(
    position=0.1,
    log_scaling=0.025,
    rotation=0.005,
    alpha_logit=0.2,
    feature=0.01
  )

  params = {k: torch.nn.Parameter(x, requires_grad=True) 
                           if k in learning_rates else x
                           for k, x in gaussians.items()}
  param_groups = [
    dict(params=[params[name]], lr=lr * base_lr, name=name)
      for name, lr in learning_rates.items()
  ]

  return Adam(param_groups), Gaussians2D(**params, batch_size=gaussians.batch_size)


def display_image(image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow('rendered', image)
    cv2.waitKey(1)
    


def main():
  device = torch.device('cuda:0')

  args = parse_args()
  
  ref_image = cv2.imread(args.image_file)
  h, w = ref_image.shape[:2]

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
          debug=args.debug, device_memory_GB=0.1, kernel_profiler=args.profile)

  print(f'Image size: {w}x{h}')

  if args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_FULLSCREEN)

  torch.manual_seed(args.seed)

  gaussians = random_2d_gaussians(args.n, (w, h)).to(torch.device('cuda:0'))
  opt, params = optimizer(gaussians, base_lr=1.0)
  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255

  config = RasterConfig(tile_size=args.tile_size)

  while True:
    if args.profile:
      ti.profiler.clear_kernel_profiler_info()

    start = time.time()

    for _ in range(args.epoch):
      opt.zero_grad()

      image = render_gaussians(params, (w, h), config)
      loss = torch.nn.functional.l1_loss(image, ref_image) 
      
      loss.backward()

      check_finite(params)
      opt.step()

      with torch.no_grad():
        params.log_scaling.clamp_(min=-1, max=6)
    
      if args.show:
        display_image(image)
      
    end = time.time()

    print(f'{args.epoch} iterations: {end - start:.3f}s at {args.epoch / (end - start):.1f} iters/sec')

    if args.profile:
      ti.profiler.print_kernel_profiler_info("count")
  

if __name__ == '__main__':
  main()