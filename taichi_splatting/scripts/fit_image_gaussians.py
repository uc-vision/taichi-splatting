import cv2
import argparse
import taichi as ti

import torch
from torch.optim import Adam
from taichi_splatting.data_types import CameraParams, Gaussians3D
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.renderer import render_gaussians
from taichi_splatting.tests.random_data import random_3d_gaussians
from taichi_splatting.torch_ops.transforms import make_homog, transform44

from taichi_splatting.torch_ops.util import check_finite
from taichi_splatting.benchmarks.util import with_timer, with_profiler



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


def optimizer(gaussians: Gaussians3D, base_lr=1.0):

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

  return Adam(param_groups), Gaussians3D(**params, batch_size=gaussians.batch_size)


def display_image(image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow('rendered', image)
    cv2.waitKey(1)
    

def orthographic_camera(w, h):
  # setup simple orthographic camera

  projection = torch.tensor([
    [w/2, 0,   w / 2],
    [0, h/2,   h / 2],
    [0, 0,     1]
  ], dtype=torch.float32)

  return CameraParams(
    image_size=(w, h),
    T_image_camera=projection,
    T_camera_world=torch.eye(4, dtype=torch.float32),
    near_plane=1.,
    far_plane=100.,
    orthographic=True
  )


def main():
  device = torch.device('cuda:0')
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

  args = parse_args()
  
  ref_image = cv2.imread(args.image_file)
  h, w = ref_image.shape[:2]

  ti.init(arch=ti.cuda, log_level=ti.DEBUG, offline_cache=True,
          debug=args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_FULLSCREEN)

  torch.manual_seed(args.seed)

  camera = orthographic_camera(w, h)

  point3d = torch.tensor([[100, 100, 4, 1]], dtype=torch.float32)
  point2d = transform44(camera.T_image_world, point3d) / point3d[:, 3:4]

  print(point2d)



  gaussians = random_3d_gaussians(args.n, camera).to(torch.device('cuda:0'))
  opt, params = optimizer(gaussians, base_lr=1.0)
  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255

  config = RasterConfig(tile_size=args.tile_size)

  def train_epoch():

    for _ in range(args.epoch):
      opt.zero_grad()

      render = render_gaussians(params.packed(), params.feature, camera, config)
      loss = torch.nn.functional.l1_loss(render.image, ref_image) 
      
      loss.backward()

      check_finite(params)
      opt.step()

      with torch.no_grad():
        params.log_scaling.clamp_(min=-1, max=6)
    
      if args.show:
        display_image(render.image)
      

  while True:
    if args.profile:
      prof = with_profiler(train_epoch)
      print(prof.key_averages().table(sort_by="self_cuda_time_total", 
                                      row_limit=25, max_name_column_width=70))
    else:
      duration = with_timer(train_epoch)
      print(f'{args.epoch} iterations: {duration:.3f}s at {args.epoch / (duration):.1f} iters/sec')




  

if __name__ == '__main__':
  main()