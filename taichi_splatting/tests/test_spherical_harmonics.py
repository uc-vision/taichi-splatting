import torch
import taichi as ti
from tqdm import tqdm
from taichi_splatting.tests.util import eval_with_grad

from taichi_splatting.torch import spherical_harmonics as torch_sh
from taichi_splatting import spherical_harmonics as taichi_sh

import warnings
warnings.filterwarnings('ignore') 


ti.init(debug=True)

def random_inputs(seed, max_dim=3, max_deg=3, max_n=5, device='cpu'):
    torch.random.manual_seed(seed)
    dimension = torch.randint(1, max_dim, (1,)).item()
    degree = torch.randint(1, max_deg + 1, (1, )).item()
    n = torch.randint(1, max_n, (1,) ).item()

    params = torch.rand(n, dimension, (degree + 1)**2, device=device, dtype=torch.float32)
    dirs = torch.randn(n, 3, device=device, dtype=torch.float32)
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    return (params, dirs)


def test_sh(iters = 100, device='cpu'):

  seeds = torch.randint(0, 10000, (iters, ), device=device)
  for seed in tqdm(seeds):
      inputs  = random_inputs(seed, device=device)

      out1, grads1 = eval_with_grad(torch_sh.evaluate_sh, *inputs)
      out2, grads2 = eval_with_grad(taichi_sh.evaluate_sh, *inputs)

      assert torch.allclose(out1, out2, atol=1e-5), f"out1 != out2 seed={seed}"
      for grad1, grad2 in zip(grads1, grads2):
        if not torch.allclose(grad1, grad2, atol=1e-5):
          raise AssertionError(f"grad1 != grad2 for seed={seed}")

def gradcheck(func, args, **kwargs):
  args = [x.requires_grad_(True) for x in args]
  torch.autograd.gradcheck(func, args, **kwargs, eps=1e-2, atol=1e-3, rtol=1e-2)

def test_sh_gradtest(iters = 100, device='cpu'):

  seeds = torch.randint(0, 10000, (iters, ), device=device)
  for seed in tqdm(seeds):
      inputs = random_inputs(seed, device=device)
      gradcheck(taichi_sh.evaluate_sh, inputs)

if __name__ == '__main__':
  test_sh()
  test_sh_gradtest()