import math
import cv2
import argparse

import torch
from torch.optim import Adam

from taichi_splatting.renderer2d import render_gaussians, Gaussians2D
import taichi as ti

from taichi_splatting.torch_ops.projection import inverse_sigmoid
from taichi_splatting.torch_ops.util import check_finite


def random_2d_gaussians(n, image_size, seed=0):
  w, h = image_size
  scale = 0.5 * w / math.sqrt(n) 

  position = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)
  depth = torch.rand(n)  
  
  # scaling = torch.full_like(position, scale)
  scaling = (torch.rand(n, 2) + 0.5) * scale 

  rotation = torch.randn(n, 2) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

  alpha = torch.rand(n) * 0.5 + 0.1

  return Gaussians2D(
    position=position,
    depth=depth,
    log_scaling=torch.log(scaling),
    rotation=rotation,
    alpha_logit=inverse_sigmoid(alpha),
    feature=torch.rand(n, 3),

    batch_size=(n,)
  )

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--n', type=int, default=20000)
  parser.add_argument('--debug', action='store_true')
  

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

  ti.init(arch=ti.cuda, log_level=ti.INFO, debug=args.debug)

  print(f'Image size: {w}x{h}')
  cv2.namedWindow('rendered', cv2.WINDOW_FULLSCREEN)

  torch.manual_seed(args.seed)

  gaussians = random_2d_gaussians(args.n, (w, h)).to(torch.device('cuda:0'))
  opt, params = optimizer(gaussians, base_lr=1.0)

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255

  while True:
  
    opt.zero_grad()

    image = render_gaussians(params, (w, h), tile_size=args.tile_size)
    loss = torch.nn.functional.l1_loss(image, ref_image) 
    
    loss.backward()

    check_finite(params)
    opt.step()

    with torch.no_grad():
      params.log_scaling.clamp_(min=-1, max=6)
  

    display_image(image)
    
  

if __name__ == '__main__':
  main()