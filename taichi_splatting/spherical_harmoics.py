from functools import cache
import math
from beartype import beartype
import taichi as ti
from taichi.math import vec3
import torch
from tqdm import tqdm

# Derived from torch-spherical-harmonics
# https://github.com/cheind/torch-spherical-harmonics


vec16f = ti.types.vector(16, ti.float32)
vec9f = ti.types.vector(9,   ti.float32)
vec4f = ti.types.vector(4,   ti.float32)
vec1f = ti.types.vector(1,   ti.float32)

@ti.func
def rsh_cart_0(xyz: vec3) -> vec1f:
    return vec1f(
        0.282094791773878
    )


@ti.func
def rsh_cart_1(xyz: vec3) -> vec4f:
    x, y, z = xyz
    return vec4f(
        0.282094791773878,
        -0.48860251190292 * y,
        0.48860251190292 * z,
        -0.48860251190292 * x,
    )

@ti.func
def rsh_cart_2(xyz: vec3) -> vec9f:
    x, y, z = xyz

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return vec9f(
        0.282094791773878,
        -0.48860251190292 * y,
        0.48860251190292 * z,
        -0.48860251190292 * x,
        1.09254843059208 * xy,
        -1.09254843059208 * yz,
        0.94617469575756 * z2 - 0.31539156525252,
        -1.09254843059208 * xz,
        0.54627421529604 * x2 - 0.54627421529604 * y2,
    )

@ti.func
def rsh_cart_3(xyz: vec3) -> vec16f:
    x, y, z = xyz.x, xyz.y, xyz.z

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return vec16f(
        0.282094791773878,
        -0.48860251190292 * y,
        0.48860251190292 * z,
        -0.48860251190292 * x,
        1.09254843059208 * xy,
        -1.09254843059208 * yz,
        0.94617469575756 * z2 - 0.31539156525252,
        -1.09254843059208 * xz,
        0.54627421529604 * x2 - 0.54627421529604 * y2,
        -0.590043589926644 * y * (3.0 * x2 - y2),
        2.89061144264055 * xy * z,
        0.304697199642977 * y * (1.5 - 7.5 * z2),
        1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
        0.304697199642977 * x * (1.5 - 7.5 * z2),
        1.44530572132028 * z * (x2 - y2),
        -0.590043589926644 * x * (x2 - 3.0 * y2),
    )        


def check_sh_degree(sh_features):
  n_sh = sh_features.shape[2]
  n = int(math.sqrt(n_sh))

  assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
  return (n - 1)


@cache
def evaluate_sh_kernel(degree:int, dimension:int=3):
  assert degree >= 0 and degree <= 3

  rsh_cart_n = [rsh_cart_0, rsh_cart_1, rsh_cart_2, rsh_cart_3]
  rsh_cart = rsh_cart_n[degree]

  param_mat = ti.types.matrix(n=dimension, m=(degree + 1)**2, dtype=ti.f32)
  vec = ti.types.vector(n=dimension, dtype=ti.f32)


  @ti.kernel    
  def _evaluate_sh3_kernel(params:ti.types.ndarray(param_mat, ndim=1), 
                          dirs:ti.types.ndarray(vec3, ndim=1), 
                          out:ti.types.ndarray(vec, ndim=1)):
      
      for i in range(params.shape[0]):
          coeffs = rsh_cart(dirs[i])
          params_i = params[i]

          for d in range(dimension):
              out[i][d] = coeffs.dot(params_i[d, :])

  return _evaluate_sh3_kernel


class _module_function(torch.autograd.Function):
  @staticmethod
  def forward(ctx, params:torch.Tensor, dirs:torch.Tensor) -> torch.Tensor:
      
      degree = check_sh_degree(params)
      ctx.kernel = evaluate_sh_kernel(degree, params.shape[1])

      out = torch.zeros(dirs.shape[0], params.shape[1], dtype=torch.float32, device=params.device)
      ctx.kernel(params, dirs, out)

      ctx.save_for_backward(params, dirs, out)
      return out

  @staticmethod
  def backward(ctx, doutput):
      params, dirs, out = ctx.saved_tensors
      out.grad = doutput

      ctx.kernel.grad(params, dirs, out)
      return params.grad, dirs.grad



@beartype
def evaluate_sh(params:torch.Tensor,  # N, K (degree + 1)^2,  (usually K=3, for RGB)
                dirs:torch.Tensor     # N, 3
                ) -> torch.Tensor:    # N, K
    
    return _module_function.apply(params.contiguous(), dirs.contiguous())


def random_inputs(max_dim=6, max_deg=3, max_n=1000, device='cpu'):
    dimension = torch.randint(1, max_dim, (1,)).item()
    degree = torch.randint(0, max_deg + 1, (1, )).item()

    n = torch.randint(1, max_n, (1,) ).item()

    params = torch.rand(n, dimension, (degree + 1)**2, device=device, dtype=torch.float32)
    dirs = torch.randn(n, 3, device=device, dtype=torch.float32)
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    return params, dirs


def test_sh(iters = 100, device='cpu'):
  from taichi_splatting.torch import spherical_harmonics as sh

  for _ in tqdm(range(iters)):  
      params, dirs = random_inputs(device=device)

      params.requires_grad_(True)
      out1 = evaluate_sh(params, dirs)


      out2 = sh.evaluate_sh(params, dirs)
      assert torch.allclose(out1, out2, atol=1e-5)




if __name__ == '__main__':
    ti.init(debug=True)

    test_sh(1000, device='cpu')

    
    