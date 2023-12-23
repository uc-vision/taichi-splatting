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
  log_scaling = torch.log(torch.rand(n, 2) * 2 * scale) 

  rotation = torch.randn(n, 2) 
  rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)

  alpha_logit = inverse_sigmoid(torch.rand(n))

  return Gaussians2D(
    position=position,
    depth=depth,
    log_scaling=log_scaling,
    rotation=rotation,
    alpha_logit=alpha_logit,
    feature=torch.rand(n, 3),

    batch_size=(n,)
  )

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)

  return parser.parse_args()



def main():
  ti.init(arch=ti.cuda)

  args = parse_args()
  image = cv2.imread(args.image_file)
  h, w = image.shape[:2]

  print(f'Image size: {w}x{h}')

  cv2.imshow('image', image)
  cv2.waitKey(0)

  gaussians = random_2d_gaussians(10000, (w, h)).to(torch.device('cuda:0'))

  image = render_gaussians(gaussians, (w, h), tile_size=16)
  image = (image * 255).to(torch.uint8)

  image = image.cpu().numpy()
  cv2.imshow('rendered', image)
  cv2.waitKey(0)
  

if __name__ == '__main__':
  main()