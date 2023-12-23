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

    return params, dirs, dict(dim=dimension, deg=degree, n=n, seed=seed)




def test_sh(iters = 100, device='cpu'):

  seeds = torch.randint(0, 10000, (iters, ), device=device)
  for seed in tqdm(seeds):
      params, dirs, args = random_inputs(seed, device=device)

      out1, grads1 = eval_with_grad(torch_sh.evaluate_sh, params, dirs)
      out2, grads2 = eval_with_grad(taichi_sh.evaluate_sh, params, dirs)

      assert torch.allclose(out1, out2, atol=1e-5), f"out1 != out2 for {args}"
      for grad1, grad2 in zip(grads1, grads2):
        assert torch.allclose(grad1, grad2, atol=1e-5), f"grad1 != grad2 for {args}"

if __name__ == '__main__':
  test_sh()
