from dataclasses import replace
from functools import partial
import math
import os
from pathlib import Path
from beartype import beartype
import cv2
import argparse
import taichi as ti

import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.renderer2d import point_basis, project_gaussians2d, uniform_split_gaussians2d

from taichi_splatting.optim.sparse_adam import SparseAdam
from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time
import torch.nn.functional as F

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--pixel_tile', type=str, help='Pixel tile for backward pass default "2,2"')

  parser.add_argument('--n', type=int, default=1000)
  parser.add_argument('--target', type=int, default=None)
  parser.add_argument('--iters', type=int, default=2000)

  parser.add_argument('--epoch', type=int, default=4, help='base epoch size (increases with t)')
  parser.add_argument('--max_epoch', type=int, default=64)

  parser.add_argument('--prune_rate', type=float, default=0.01, help='Rate of pruning proportional to number of points')
  parser.add_argument('--opacity_reg', type=float, default=0.0001)
  parser.add_argument('--scale_reg', type=float, default=100.0)

  parser.add_argument('--threaded', action='store_true', help='Use taichi dedicated thread')

  parser.add_argument('--antialias', action='store_true')

  parser.add_argument('--write_frames', type=Path, default=None)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  
  args = parser.parse_args()

  if args.pixel_tile:
    args.pixel_tile = tuple(map(int, args.pixel_tile.split(',')))

  return args


def log_lerp(t, a, b):
  return math.exp(math.log(b) * t + math.log(a) * (1 - t))

def lerp(t, a, b):
  return b * t + a * (1 - t)


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  

def train_epoch(opt:SparseAdam, params:ParameterClass, ref_image, 
        config:RasterConfig,        
        epoch_size=100, 
        grad_alpha=0.9, 
        opacity_reg=0.0,
        scale_reg=0.0):
    
  h, w = ref_image.shape[:2]

  point_heuristics = torch.zeros((params.batch_size[0], 3), device=params.position.device)

  for i in range(epoch_size):
    opt.zero_grad()

    with torch.enable_grad():
      gaussians = Gaussians2D.from_tensordict(params.tensors)
      gaussians2d = project_gaussians2d(gaussians)  

      raster = rasterize(gaussians2d=gaussians2d, 
        depth=gaussians.z_depth.clamp(0, 1),
        features=gaussians.feature, 
        image_size=(w, h), 
        config=config)
      
  


      scale = torch.exp(gaussians.log_scaling) / min(w, h)
      loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
              + opacity_reg * gaussians.opacity.mean()
              + scale_reg * scale.pow(2).mean())

      loss.backward()


    check_finite(gaussians, 'gaussians', warn=True)
    visible = torch.nonzero(raster.point_heuristics[:, 0]).squeeze(1)

    opt.step(visible_indexes = visible, basis=point_basis(gaussians[visible]))

    point_heuristics =  raster.point_heuristics if i == 0 \
        else (1 - grad_alpha) * point_heuristics + grad_alpha * raster.point_heuristics
      
  return raster.image, point_heuristics


def make_epochs(total_iters, first_epoch, max_epoch):
  iteration = 0
  epochs = []
  while iteration < total_iters:

    t = iteration / total_iters
    epoch_size = math.ceil(log_lerp(t, first_epoch, max_epoch))

    if iteration + epoch_size * 2 > total_iters:
      # last epoch can just use the extra iterations
      epoch_size = total_iters - iteration

    iteration += epoch_size
    epochs.append(epoch_size)

  return epochs


@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask

def randomize_n(t:torch.Tensor, n:int):
  """ Randomly select n of the largest values in a tensor using torch.multinomial"""
  probs = F.normalize(t, dim=0)
  mask = torch.zeros_like(t, dtype=torch.bool)

  if n > 0:
    selected_indices = torch.multinomial(probs, n, replacement=False)
    mask[selected_indices] = True

  return mask
  
def find_split_prune(n, target, n_prune, point_heuristics):
    prune_cost, densify_score = point_heuristics.unbind(dim=1)
    
    prune_mask = take_n(prune_cost, n_prune, descending=False)

    target_split = ((target - n) + n_prune) 
    # split_mask = randomize_n(densify_score, min(target_split, n))
    split_mask = take_n(densify_score, target_split, descending=True)

    both = (split_mask & prune_mask)
    return split_mask ^ both, prune_mask ^ both

def split_prune(params:ParameterClass, t, target, prune_rate, point_heuristics):
  n = params.batch_size[0]

  split_mask, prune_mask = find_split_prune(n = n, 
                  target = target,
                  # n_prune=int(prune_rate * n * (1 - t)),
                  n_prune=int(prune_rate * n),
                  point_heuristics=point_heuristics)

  to_split = params[split_mask]

  
  splits = uniform_split_gaussians2d(Gaussians2D.from_tensordict(to_split.tensors), random_axis=True)

  params = params[~(split_mask | prune_mask)]
  params = params.append_tensors(splits.to_tensordict())
  params.replace(rotation = torch.nn.functional.normalize(params.rotation.detach()))

  return params, dict(      
    split = split_mask.sum().item(),
    prune = prune_mask.sum().item()
  )



def main():
  torch.set_printoptions(precision=4, sci_mode=False)

  cmd_args = parse_args()
  device = torch.device('cuda:0')

  torch.set_grad_enabled(False)

  
  ref_image = cv2.imread(cmd_args.image_file)
  assert ref_image is not None, f'Could not read {cmd_args.image_file}'

  h, w = ref_image.shape[:2]


  TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=cmd_args.debug, device_memory_GB=0.1, threaded=cmd_args.threaded)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)
  lr_range = (1.0, 0.05)

  torch.manual_seed(cmd_args.seed)
  torch.cuda.random.manual_seed(cmd_args.seed)
  gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(torch.device('cuda:0'))
  
  parameter_groups = dict(
    position=dict(lr=lr_range[0], type='local_vector'),
    log_scaling=dict(lr=0.025),

    rotation=dict(lr=0.25),
    alpha_logit=dict(lr=0.2),
    feature=dict(lr=0.03, type='vector')
  )

  create_optimizer = partial(SparseAdam, betas=(0.9, 0.95))


  params = ParameterClass(gaussians.to_tensordict(), 
        parameter_groups, optimizer=create_optimizer)
  


  keys = set(params.keys())
  trainable = set(params.optimized_keys())

  print(f'attributes - trainable: {trainable} other: {keys - trainable}')

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
  
  config = RasterConfig(compute_point_heuristics=True,
                        tile_size=cmd_args.tile_size, 
                        gaussian_scale=3.0, 
                        blur_cov=0.3 if not cmd_args.antialias else 0.0,
                        antialias=cmd_args.antialias,
                        pixel_stride=cmd_args.pixel_tile or (2, 2))

  



  def timed_epoch(*args, **kwargs):
    start = time.time()
    image, point_heuristics = train_epoch(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    return image, point_heuristics, end - start


  train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch
  epochs = make_epochs(cmd_args.iters, cmd_args.epoch, cmd_args.max_epoch)

  pbar = tqdm(total=cmd_args.iters)
  iteration = 0
  for  epoch_size in epochs:

    t = (iteration + epoch_size * 0.5) / cmd_args.iters
    params.set_learning_rate(position = log_lerp(t, *lr_range))
    metrics = {}

    image, point_heuristics, epoch_time = train(params.optimizer, params, ref_image, 
                                      epoch_size=epoch_size, config=config, 
                                      opacity_reg=cmd_args.opacity_reg,
                                      scale_reg=cmd_args.scale_reg)


    if cmd_args.show:
      display_image('rendered', image)

  
    if cmd_args.write_frames:
      filename = cmd_args.write_frames / f'{iteration:04d}.png'
      filename.parent.mkdir(exist_ok=True, parents=True)
      print(f'Writing {filename}')
      cv2.imwrite(str(filename), 
                  (image.detach().clamp(0, 1) * 255).cpu().numpy())

    metrics['CPSNR'] = psnr(ref_image, image).item()
    metrics['n'] = params.batch_size[0]


    if cmd_args.target and iteration + epoch_size < cmd_args.iters:
      t_points = min(math.pow(t * 2, 0.5), 1.0)
      target = math.ceil(params.batch_size[0] * (1 - t_points) + t_points * cmd_args.target)
      params, prune_metrics = split_prune(params, t, target, cmd_args.prune_rate, point_heuristics)
      metrics.update(prune_metrics)

    for k, v in metrics.items():
      if isinstance(v, float):
        metrics[k] = f'{v:.2f}'
      if isinstance(v, int):
        metrics[k] = f'{v:4d}'

    pbar.set_postfix(**metrics)

    iteration += epoch_size
    pbar.update(epoch_size)

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
      
  main()