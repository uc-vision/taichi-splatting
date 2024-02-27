

from functools import partial

from tqdm import tqdm
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.misc.encode_depth import encode_depth32
from taichi_splatting.misc.renderer2d import  project_gaussians2d
from taichi_splatting.rasterizer.function import rasterize_with_tiles
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.mapper.tile_mapper import map_to_tiles

import torch
import taichi as ti

import cv2

def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(0)


def make_inputs(seed, device=torch.device('cuda:0')):
    torch.random.manual_seed(seed)

    # n = torch.randint(1, 2, (1, ), device=device).item()
    image_size = (8, 8)
    config = RasterConfig(tile_size=4, pixel_stride=(1, 1))

    gaussians = random_2d_gaussians(1, image_size, scale_factor=0.1, alpha_range=(0.1, 0.5)).to(device=device)  
    gaussians2d = project_gaussians2d(gaussians)

    overlap_to_point, tile_ranges = map_to_tiles(gaussians2d, encode_depth32(gaussians.depth), image_size, config)
    gaussians2d, colors = [x.to(dtype=torch.float64) for x in [gaussians2d, gaussians.feature]]

    def rasterize(uv, conic, alpha, colors):
      packed = torch.cat([uv, conic, alpha], dim=-1)
      return rasterize_with_tiles(packed, colors, overlap_to_point=overlap_to_point, tile_overlap_ranges=tile_ranges.view(-1, 2), 
                     image_size=image_size, config=config).image
    
    return (gaussians2d[:, 0:2].requires_grad_(False), 
            gaussians2d[:, 2:5].requires_grad_(False), 
            gaussians2d[:, 5:6].requires_grad_(True), 
            colors.requires_grad_(False)), rasterize




def test_rasterizer_gradcheck(iters = 100, device=torch.device('cuda:0')):
  torch.random.manual_seed(0)
  seeds = torch.randint(0, 10000, (iters, ), device=device)

  for seed in tqdm(seeds, desc="rasterizer_gradcheck"):
      inputs, render = make_inputs(seed)
      torch.autograd.gradcheck(render, inputs, eps=1e-6, check_grad_dtypes=True, check_undefined_grad=True)

def main():
  torch.set_printoptions(precision=10, sci_mode=False)

  ti.init(arch=ti.cuda, default_fp=ti.f64)
  test_rasterizer_gradcheck()


if __name__ == "__main__":
    main()