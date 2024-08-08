from dataclasses import replace
from functools import cache
import math
import taichi as ti
import torch

from taichi_splatting.data_types import Gaussians2D
from taichi_splatting.torch_lib.projection import inverse_sigmoid


@ti.kernel
def split_kernel(
  indexes:ti.types.ndarray(dtype=ti.int64, ndim=1),
  n : ti.types.ndarray(dtype=ti.i64, ndim=1),

  opacity:ti.types.ndarray(ndim=1), 
  binomial_table:ti.types.ndarray(dtype=ti.f32, ndim=2), 

  out_opacity:ti.types.ndarray(ndim=1), 
  out_scale:ti.types.ndarray(ti.f32, ndim=1)):
  
  for i in range(indexes):
    idx = indexes[i]

    n = ti.min(n[idx], binomial_table.shape[0] - 1)
    denom_sum = 0.0

    new_opacity = 1.0 - ti.pow(1. - opacity, 1. / n)

    for j in range(n):
      for k in range(j):
        bin_coeff = binomial_table[i, k]
        
        term = (ti.pow(-1, k) / ti.sqrt(k + 1)) * ti.pow(new_opacity, k + 1)
        denom_sum += (bin_coeff * term)

      out_scale[i] = (opacity[idx] / denom_sum)
      out_opacity[i] = new_opacity

@cache 
def binomial_table(max_split:int, device:torch.device):
  table = torch.zeros((max_split, max_split), dtype=torch.float32, device=device)
  for n in range(max_split):
    for k in range(n+1):
        table[n, k] = math.comb(n, k)

def sample_gaussians(probs, n):
    probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    indexes = torch.multinomial(probs, n, replacement=True)
    counts = torch.bincount(indexes, minlength=probs.shape[0]).unsqueeze(-1)
    return indexes, counts

def compute_split2d(gaussians:Gaussians2D, probs, n, max_split=8):

  binomials = binomial_table(max_split, gaussians.device)
  indexes, counts = sample_gaussians(probs, n)

  opacity = torch.empty((n, 1), dtype=torch.float32, device=gaussians.device)
  scale_factor = torch.empty_like(opacity)
  split_kernel(indexes, counts, gaussians.opacity, binomials, opacity, scale_factor)

  splits:Gaussians2D = gaussians[indexes]
  splits = replace(splits, 
                   alpha_logit=inverse_sigmoid(opacity), 
                   log_scaling = torch.log(scale_factor * splits.scaling))

  return splits