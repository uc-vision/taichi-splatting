from functools import partial
import math
from pathlib import Path
from beartype import beartype
import cv2
import argparse
import numpy as np
import taichi as ti

import torch
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.encode_depth import encode_depth32
from taichi_splatting.misc.renderer2d import project_gaussians2d, sample_gaussians, split_gaussians2d, uniform_split_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_ops.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time
from torch import optim

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--pixel_tile', type=str, help='Pixel tile for backward pass default "2,2"')

  parser.add_argument('--n', type=int, default=1000)
  parser.add_argument('--target', type=int, default=None)
  parser.add_argument('--max_epoch', type=int, default=100)
  parser.add_argument('--split_rate', type=float, default=0.5, help='Rate of pruning proportional to number of points')
  parser.add_argument('--opacity_reg', type=float, default=0.0001)
  parser.add_argument('--scale_reg', type=float, default=0.0001)

  parser.add_argument('--write_frames', type=Path, default=None)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--epoch_size', type=int, default=20, help='Number of iterations per measurement/profiling')
  
  args = parser.parse_args()

  if args.pixel_tile:
    args.pixel_tile = tuple(map(int, args.pixel_tile.split(',')))

  return args




def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  

def train_epoch(opt, gaussians, ref_image, epoch_size=100, 
        config:RasterConfig = RasterConfig(), grad_alpha=0.9, 
        opacity_reg=0.0,
        scale_reg=0.0,
        noise_threshold=0.05,
        noise_lr=100.0, 
        k = 100):
    
    h, w = ref_image.shape[:2]

    split_heuristics = torch.zeros((gaussians.batch_size[0], 2), device=gaussians.position.device)

    for i in range(epoch_size):
      opt.zero_grad()

      gaussians2d = project_gaussians2d(gaussians)  
      depths = encode_depth32(gaussians.z_depth)

      opacity = torch.sigmoid(gaussians.alpha_logit).unsqueeze(-1)


      raster = rasterize(gaussians2d=gaussians2d, 
        encoded_depths=depths,
        features=gaussians.feature, 
        image_size=(w, h), 
        config=config,
        compute_split_heuristics=True)


      scale = torch.exp(gaussians.log_scaling)
      loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
              + opacity_reg * opacity.mean()
              + scale_reg * scale.mean())

      loss.backward()

      check_finite(gaussians, 'gaussians', warn=True)
      opt.step()

      with torch.no_grad():
        gaussians.log_scaling.clamp_(min=-1, max=4)

        split_heuristics =  raster.point_split_heuristics if i == 0 \
            else (1 - grad_alpha) * split_heuristics + grad_alpha * raster.point_split_heuristics
        
        opacity = torch.sigmoid(gaussians.alpha_logit)
        op_factor = torch.sigmoid(k * (noise_threshold - opacity)).unsqueeze(1)

        noise = sample_gaussians(gaussians) * op_factor * noise_lr
        gaussians.position += noise



      prune_cost, densify_score = split_heuristics.unbind(dim=1)
    return raster.image, prune_cost, densify_score 


def main():
  device = torch.device('cuda:0')

  cmd_args = parse_args()
  
  ref_image = cv2.imread(cmd_args.image_file)
  assert ref_image is not None, f'Could not read {cmd_args.image_file}'

  h, w = ref_image.shape[:2]

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
          debug=cmd_args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)
    cv2.namedWindow('err', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('gradient', w, h)
    cv2.resizeWindow('err', w, h)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), scale_factor=0.1).to(torch.device('cuda:0'))
  learning_rates = dict(
    position=0.1,
    log_scaling=0.05,
    rotation=0.005,
    alpha_logit=0.1,
    feature=0.01
  )
  create_optimizer = partial(optim.Adam, foreach=True, betas=(0.7, 0.999), amsgrad=True, weight_decay=0.0)


  params = ParameterClass.create(gaussians.to_tensordict(), learning_rates, base_lr=1.0, optimizer=create_optimizer)
  keys = set(params.keys())
  trainable = set(params.optimized_keys())

  print(f'attributes - trainable: {trainable} other: {keys - trainable}')

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
  
  config = RasterConfig(tile_size=cmd_args.tile_size, gaussian_scale=3.0, pixel_stride=cmd_args.pixel_tile or (2, 2))

  @beartype
  def take_n(t:torch.Tensor, n:int, descending=False):
    """ Return mask of n largest or smallest values in a tensor."""
    idx = torch.argsort(t, descending=descending)[:n]

    # convert to mask
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask[idx] = True

    return mask
    
  def split_prune(n, target, n_prune, densify_score, prune_cost):
      prune_mask = take_n(prune_cost, n_prune, descending=False)

      target_split = ((target - n) + n_prune) 
      split_mask = take_n(densify_score, target_split, descending=True)

      both = (split_mask & prune_mask)
      return split_mask ^ both, prune_mask ^ both




  def timed_epoch(*args, **kwargs):
    start = time.time()
    image, grad, vis = train_epoch(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    return image, grad, vis, end - start


  train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch

  
  for epoch in range(cmd_args.max_epoch):
    epoch_size = cmd_args.epoch_size

    image, densify_score, prune_cost, epoch_time = train(params.optimizer, params, ref_image, 
                                        epoch_size=epoch_size, config=config, 
                                        opacity_reg=cmd_args.opacity_reg,
                                        scale_reg=cmd_args.scale_reg)
    

    with torch.no_grad():

      if cmd_args.show:
        gaussians2d = project_gaussians2d(params)
        depths = encode_depth32(params.z_depth)
        raster =  rasterize(gaussians2d, depths, densify_score.contiguous().unsqueeze(-1), 
                            image_size=(w, h), config=config, compute_split_heuristics=True)

        err = torch.abs(ref_image - image)
        
        display_image('gradient', (0.5 * raster.image / raster.image.mean() ))
        display_image('rendered', image)
        display_image('err',  0.25 * err / err.mean(dim=(0, 1), keepdim=True))

    
      if cmd_args.write_frames:
        filename = cmd_args.write_frames / f'{epoch:04d}.png'
        filename.parent.mkdir(exist_ok=True, parents=True)
        print(f'Writing {filename}')
        cv2.imwrite(str(filename), 
                    (image.detach().clamp(0, 1) * 255).cpu().numpy())

      cpsnr = psnr(ref_image, image)
      print(f'{epoch + 1}: {epoch_size / epoch_time:.1f} iters/sec CPSNR {cpsnr:.2f}')

      if cmd_args.target and epoch < cmd_args.max_epoch - 1:
        gaussians = Gaussians2D(**params.tensors, batch_size=params.batch_size)

        t = (epoch + 1) / (cmd_args.max_epoch - 1)
        t_points = min(math.pow(t * 2 , 0.5), 1.0)

        n = gaussians.batch_size[0]

        split_mask, prune_mask = split_prune(n = n, target = math.ceil(cmd_args.n * (1 - t_points) + t_points * cmd_args.target),
                    n_prune=int(cmd_args.split_rate * n * (1 - t)**2),
                    densify_score=densify_score, prune_cost=prune_cost)

        if prune_mask.sum() > 0:
          print(f"thresholds: split {densify_score[split_mask].min()} prune {prune_cost[prune_mask].max()}")


        splits = uniform_split_gaussians2d(gaussians[split_mask], noise=0.1)


        params = params[~(split_mask | prune_mask)]
        params = params.append_tensors(splits.to_tensordict())

        print(f" split {split_mask.sum()}, pruned {prune_mask.sum()} {params.batch_size} points")

        


def with_benchmark(f):
  def g(*args , **kwargs):
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
      with record_function("model_inference"):
        result = f(*args, **kwargs)
        torch.cuda.synchronize()

      prof_table = prof.key_averages().table(sort_by="self_cuda_time_total", 
                                          row_limit=25, max_name_column_width=100)
      print(prof_table)
      return result
  return g

  

if __name__ == '__main__':
  main()