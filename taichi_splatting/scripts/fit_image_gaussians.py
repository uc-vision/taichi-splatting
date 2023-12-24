import math
import cv2
import argparse

import torch
from torch.optim import Adam

from taichi_splatting.renderer2d import render_gaussians, Gaussians2D
import taichi as ti

def inverse_sigmoid(x:torch.Tensor):
  return torch.log(x / (1 - x))


def random_2d_gaussians(n, image_size):
  w, h = image_size

  scale = w / math.sqrt(n) 

  position = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)
  depth = torch.rand(n)  
  scaling = (torch.rand(n, 2) + 0.5)  * scale

  rotation = torch.randn(n, 2) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

  alpha = torch.rand(n) * 0.5 + 0.499

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

  return parser.parse_args()


def optimizer(gaussians: Gaussians2D):

  learning_rates = dict(
    position=0.00016,
    log_scaling=0.005,
    rotation=0.001,
    alpha_logit=0.05,
    feature=0.0025
  )

  params = {k: torch.nn.Parameter(x, requires_grad=True) 
                           if k in learning_rates else x
                           for k, x in gaussians.items()}


  param_groups = [
    dict(params=[params[name]], lr=param, name=name)
      for name, param in learning_rates.items()
  ]

  return Adam(param_groups), Gaussians2D(**params, batch_size=gaussians.batch_size)


def display_image(image):
    image = (image * 255).to(torch.uint8)
    image = image.cpu().numpy()
    cv2.imshow('rendered', image)
    cv2.waitKey(1)
    

def main():
  ti.init(arch=ti.cuda, log_level=ti.DEBUG)
  device = torch.device('cuda:0')

  args = parse_args()
  ref_image = cv2.imread(args.image_file)
  h, w = ref_image.shape[:2]

  print(f'Image size: {w}x{h}')
  cv2.namedWindow('rendered', cv2.WINDOW_GUI_EXPANDED)


  gaussians = random_2d_gaussians(100, (w, h)).to(torch.device('cuda:0'))
  opt, params = optimizer(gaussians)

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255

  while True:
    opt.zero_grad()

    image = render_gaussians(params, (w, h), tile_size=16)
    loss = torch.nn.functional.l1_loss(image, ref_image)

    loss.backward()
    opt.step()

    display_image(image)
    
  

if __name__ == '__main__':
  main()