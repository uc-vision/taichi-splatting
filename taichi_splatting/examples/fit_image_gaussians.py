import math
import cv2
import argparse
import taichi as ti

import torch
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.encode_depth import encode_depth32
from taichi_splatting.misc.renderer2d import project_gaussians2d, split_gaussians2d, uniform_split_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_ops.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--n', type=int, default=20000)


  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--split', action='store_true', help="Enable splitting of gaussians")

  parser.add_argument('--grad-threshold', type=float, default=5e-5)
  parser.add_argument('--size-threshold', type=float, default=8)

  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--epoch', type=int, default=100, help='Number of iterations per measurement/profiling')
  
  return parser.parse_args()

def inverse_sigmoid(x):
  return math.log(x / (1 - x))


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  


def train_epoch(opt, gaussians, ref_image, epoch_size=100, config:RasterConfig = RasterConfig()):
    h, w = ref_image.shape[:2]

    gradient = torch.zeros(gaussians.batch_size, device=gaussians.position.device)

    for _ in range(epoch_size):
      opt.zero_grad()

      gaussians2d = project_gaussians2d(gaussians)
      gaussians2d.retain_grad()

      gaussians.feature.retain_grad()
      
      depths = encode_depth32(gaussians.depth)
      image, alpha = rasterize(gaussians2d=gaussians2d, 
        encoded_depths=depths,
        features=gaussians.feature, 
        image_size=(w, h), 
        config=config)

      loss = torch.nn.functional.l1_loss(image, ref_image) 
      loss.backward()

      check_finite(gaussians)
      opt.step()

      with torch.no_grad():
        gaussians.log_scaling.clamp_(min=-1, max=4)
        gradient += gaussians2d.grad[:, 0:2].norm(dim=-1) 

    return image, gradient


def main():
  device = torch.device('cuda:0')

  cmd_args = parse_args()
  
  ref_image = cv2.imread(cmd_args.image_file)
  h, w = ref_image.shape[:2]

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
          debug=cmd_args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
    cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)


  torch.manual_seed(cmd_args.seed)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), scale_factor=0.1).to(torch.device('cuda:0'))
  learning_rates = dict(
    position=0.1,
    log_scaling=0.025,
    rotation=0.005,
    alpha_logit=0.1,
    feature=0.01
  )
  

  params = ParameterClass.create(gaussians.to_tensordict(), learning_rates, base_lr=1.0)

  print(list(params.keys()))

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
  config = RasterConfig(tile_size=cmd_args.tile_size, gaussian_scale=3.)



  def timed_epoch(*args, **kwargs):
    start = time.time()
    image, grad = train_epoch(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    cpsnr = psnr(ref_image, image)
    print(f'{cmd_args.epoch / (end - start):.1f} iters/sec CPSNR {cpsnr:.2f}')
    return image, grad


  train = with_benchmark(train_epoch) if cmd_args.profile else timed_epoch

  while True:
    image, gradient = train(params.optimizer, params, ref_image, epoch_size=cmd_args.epoch, config=config)
    with torch.no_grad():

      if cmd_args.show:
        gaussians2d = project_gaussians2d(params)
        depths = encode_depth32(params.depth)
        grad_image, _ =  rasterize(gaussians2d, depths, gradient.unsqueeze(-1), image_size=(w, h), config=config)
        display_image('gradient', (0.5 * grad_image / cmd_args.grad_threshold))
        display_image('rendered', image)


      if cmd_args.split:
        gaussians = Gaussians2D(**params.tensors, batch_size=params.batch_size)

        transparent = (params.alpha_logit < inverse_sigmoid(0.01))
        split_mask = (gradient > cmd_args.grad_threshold
          ) & torch.any(gaussians.log_scaling > math.log(cmd_args.size_threshold), dim=-1)
        
        splits = split_gaussians2d(gaussians[split_mask], 
                scaling=0.8)

        params = params[~(split_mask | transparent)]
        params = params.append_tensors(splits.to_tensordict())

        print(f"Split {split_mask.sum()}, pruned {transparent.sum()} total {params.batch_size} points") 


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