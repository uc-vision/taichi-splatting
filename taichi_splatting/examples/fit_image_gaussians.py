import math
from typing import List
import cv2
import argparse
import numpy as np
import torch.nn as nn
import taichi as ti

import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite

torch.set_float32_matmul_precision('high')

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)

  parser.add_argument('--n', type=int, default=20)
  parser.add_argument('--iters', type=int, default=20)

  parser.add_argument('--epoch', type=int, default=20, help='base epoch size (increases with t)')

  parser.add_argument('--opacity_reg', type=float, default=0.0000)
  parser.add_argument('--scale_reg', type=float, default=0.0)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  
  args = parser.parse_args()

  return args


def log_lerp(t, a, b):
  return math.exp(math.log(b) * t + math.log(a) * (1 - t))

def lerp(t, a, b):
  return b * t + a * (1 - t)


def display_image(name, image):
    image = (image.clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  


def cat_values(dicts, dim=1):
  assert all(d.batch_size == dicts[0].batch_size for d in dicts)
  
  dicts = [d.to_tensordict() for d in dicts]
  assert all(d.keys() == dicts[0].keys() for d in dicts)

  keys, cls = dicts[0].keys(), type(dicts[0])
  concatenated = {k: torch.cat([d[k] for d in dicts], dim=dim) for k in keys}
  return cls.from_dict(concatenated, batch_size=dicts[0].batch_size)

def element_sizes(t):
  """ Get non batch sizes from a tensorclass"""
  return {k:v.shape[1:] for k, v in t.items()}


def split_tensorclass(t, flat_tensor:torch.Tensor):
  sizes = element_sizes(t)
  splits = [np.prod(s) for s in sizes.values()]

  tensors = torch.split(flat_tensor, splits, dim=1)
  
  return t.__class__.from_dict(
      {k: v.view(t.batch_size + s) 
       for k, v, s in zip(sizes.keys(), tensors, sizes.values())},
      batch_size=t.batch_size
  )

def flatten_tensorclass(t):
  flat_tensor = torch.cat([v.view(v.shape[0], -1) for v in t.values()], dim=1)
  return flat_tensor


class Trainer:
  def __init__(self, 
               optimizer_mlp:torch.nn.Module, 
               mlp_opt:torch.optim.Optimizer, 
               ref_image:torch.Tensor,
               config:RasterConfig, 
               opacity_reg=0.0, 
               scale_reg=0.0):
    
    self.optimizer_mlp = optimizer_mlp
    self.mlp_opt = mlp_opt

    self.config = config
    self.opacity_reg = opacity_reg
    self.scale_reg = scale_reg

    self.ref_image:torch.Tensor = ref_image
    self.running_scales = None

    
  def render(self, gaussians):
    h, w = self.ref_image.shape[:2]
    
    gaussians2d = project_gaussians2d(gaussians) 
    raster = rasterize(gaussians2d=gaussians2d, 
      depth=gaussians.z_depth.clamp(0, 1),
      features=gaussians.feature, 
      image_size=(w, h), 
      config=self.config)
    return raster

  def render_step(self, gaussians):
    with torch.enable_grad():
      raster = self.render(gaussians)

      h, w = self.ref_image.shape[:2]
      scale = torch.exp(gaussians.log_scaling) / min(w, h)
      loss = (torch.nn.functional.l1_loss(raster.image, self.ref_image) 
              + self.opacity_reg * gaussians.opacity.mean()
              + self.scale_reg * scale.pow(2).mean()
              + 0.0 * gaussians.z_depth.sum()
              )
      
      loss.backward()
      return loss.item()

  def get_gradients(self, gaussians):
    gaussians = gaussians.clone()
    gaussians.requires_grad_(True)
    self.render_step(gaussians)
    grad =  gaussians.grad
    


    scales = dict(position = 0.0002, log_scaling = 0.005, rotation = 0.0005, feature = 0.001)

    for k, v in grad.items():
      if k in scales:
        v /= (scales[k] / 1000.)

    return grad

    # mean_abs_grad = grad.abs().mean(dim=0)
    # if self.running_scales is None:
    #   self.running_scales = mean_abs_grad
    # else:
    #   self.running_scales = lerp(0.9, self.running_scales, mean_abs_grad)
      
    # return grad / (self.running_scales.unsqueeze(0) + 1e-12)
  


  def train_epoch(self, gaussians, step_size=0.01, epoch_size=100):
    for i in range(epoch_size):

      grad = self.get_gradients(gaussians)
      check_finite(grad, "grad")
      self.mlp_opt.zero_grad()

      inputs = flatten_tensorclass(grad)

      with torch.enable_grad():
        step = self.optimizer_mlp(inputs)
        step = split_tensorclass(gaussians, step)

        loss = self.render_step(gaussians - step)
        print(f'{i:3d}: {loss:.4f}')

      self.mlp_opt.step()
      gaussians = gaussians - step * step_size 

    return gaussians



def mlp(inputs, outputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity, output_activation=None):
  def linear(in_features, out_features, init_std=None):
    m = nn.Linear(in_features, out_features, bias=True)

    if init_std is not None:
      m.weight.data.normal_(0, init_std)

    m.bias.data.zero_()
    return m

  return nn.Sequential(
    
    linear(inputs, hidden_channels[0]), 
    activation(),
    norm(hidden_channels[0]),

    *[nn.Sequential(linear(hidden_channels[i], hidden_channels[i+1]), activation(), norm(hidden_channels[i+1]))
      for i in range(len(hidden_channels) - 1)],
    linear(hidden_channels[-1], outputs, init_std=1e-10),

    output_activation() if output_activation is not None else nn.Identity()
  )   


def main():
  torch.set_printoptions(precision=4, sci_mode=True)

  cmd_args = parse_args()
  device = torch.device('cuda:0')

  torch.set_grad_enabled(False)

  
  ref_image = cv2.imread(cmd_args.image_file)
  assert ref_image is not None, f'Could not read {cmd_args.image_file}'

  h, w = ref_image.shape[:2]


  TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=cmd_args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)
  torch.cuda.random.manual_seed(cmd_args.seed)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(torch.device('cuda:0')) 
  channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])


  # Create the MLP
  hidden_channels = [256, 256, 256, channels]  # Hidden layers 
  point_optimizer_mlp = mlp(inputs = channels, outputs=channels, 
              hidden_channels=hidden_channels, 
              activation=nn.ReLU,
              norm=nn.LayerNorm
              )
  point_optimizer_mlp.to(device=device)

  point_optimizer_mlp = torch.compile(point_optimizer_mlp)
  mlp_opt = torch.optim.Adam(point_optimizer_mlp.parameters(), lr=0.0001)

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255 
  config = RasterConfig()

  trainer = Trainer(point_optimizer_mlp, mlp_opt, ref_image, config, 
                    opacity_reg=cmd_args.opacity_reg, scale_reg=cmd_args.scale_reg)
  
  epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]

  pbar = tqdm(total=cmd_args.iters)
  iteration = 0
  for  epoch_size in epochs:

    metrics = {}

    # Set warmup schedule for first iterations - log interpolate 
    step_size = log_lerp(min(iteration / 1000., 1.0), 1.0, 1.0)
    
    gaussians = trainer.train_epoch(gaussians, epoch_size=epoch_size, step_size=step_size)

    image = trainer.render(gaussians).image
    if cmd_args.show:
      display_image('rendered', image)

  
    metrics['CPSNR'] = psnr(ref_image, image).item()
    metrics['n'] = gaussians.batch_size[0]


    for k, v in metrics.items():
      if isinstance(v, float):
        metrics[k] = f'{v:.2f}'
      if isinstance(v, int):
        metrics[k] = f'{v:4d}'

    pbar.set_postfix(**metrics)

    iteration += epoch_size
    pbar.update(epoch_size)


      
if __name__ == "__main__":
  main()