import torch
from tqdm import tqdm
from taichi_splatting.tests.util import compare_with_grad

from taichi_splatting.torch_lib import spherical_harmonics as torch_sh
from taichi_splatting import spherical_harmonics as slang_sh

import warnings
warnings.filterwarnings('ignore') 



def random_inputs(max_deg=3, max_n=100, device='cuda', dtype=torch.float32):
  def f(seed):
    torch.random.manual_seed(seed)
    # degree = torch.randint(1, max_deg + 1, (1, )).item()
    degree = 3
    n = torch.randint(1, max_n + 2, (1,) ).item()

    params = torch.rand( (n, 3, (degree + 1)**2), device=device, dtype=dtype)
    points = torch.randn( (n, 3), device=device, dtype=dtype)
    camera_pos = torch.randn(3, device=device, dtype=dtype)

    return (params.requires_grad_(True), points.requires_grad_(True), 
            camera_pos.requires_grad_(True))
  return f

def test_sh(iters = 100, device='cuda'):

  make_inputs = random_inputs(max_n=100, device=device, dtype=torch.float32)
  compare_with_grad("spherical_harmonics", 
    input_names=["params", "dirs", "camera_pos"], 
    output_names="out",
    f1=slang_sh.evaluate_sh_at,
    f2=torch_sh.evaluate_sh_at,
    gen_inputs=make_inputs,
    iters=iters)
  


def gradcheck(func, args, **kwargs):
  args = [x.detach().requires_grad_(x.requires_grad) for x in args]
  torch.autograd.gradcheck(func, args, **kwargs)

def test_sh_gradcheck(iters = 100, device='cuda'):

  make_inputs = random_inputs(max_n=10, device=device, dtype=torch.double)

  seeds = torch.randint(40, 10000, (iters, ), device=device)
  for seed in tqdm(seeds, desc="spherical_harmonics_gradcheck"):
      inputs = make_inputs(seed)
      gradcheck(slang_sh.evaluate_sh_at, inputs)


if __name__ == '__main__':
  

  test_sh()
  

  # slang autograd doesn't seem to work with
  # test_sh_gradcheck()