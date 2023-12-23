import torch

import taichi as ti
from taichi.math import vec2, vec3

from taichi_splatting.ti.covariance import conic_pdf_with_grad, conic_pdf
from taichi_splatting.tests.util import eval_with_grad

import warnings
warnings.filterwarnings('ignore') 


ti.init(debug=True)

def torch_conic_pdf(xy:torch.Tensor, uv:torch.Tensor, uv_conic:torch.Tensor) -> torch.Tensor:
    dx, dy = (xy - uv).T
    a, b, c = uv_conic.T
    return torch.exp(-0.5 * (dx**2 * a + dy**2 * c) - dx * dy * b)



@ti.kernel
def kernel_conic_pdf_grad(
   xy : ti.types.ndarray(vec2, ndim=1),
   uv : ti.types.ndarray(vec2, ndim=1),
   uv_conic : ti.types.ndarray(vec3, ndim=1),

   dp_duv : ti.types.ndarray(vec2, ndim=1),
   dp_dconic : ti.types.ndarray(vec3, ndim=1)):

   for i in range(uv.shape[0]):
      _, grad_uv, grad_conic = conic_pdf_with_grad(xy[i], uv[i], uv_conic[i])
      dp_duv[i] = grad_uv
      dp_dconic[i] = grad_conic

@ti.kernel
def kernel_conic_pdf(
   xy : ti.types.ndarray(vec2, ndim=1),
   uv : ti.types.ndarray(vec2, ndim=1),
   uv_conic : ti.types.ndarray(vec3, ndim=1),

   out_p : ti.types.ndarray(ti.f32, ndim=1)):

   for i in range(uv.shape[0]):
      p = conic_pdf(xy[i], uv[i], uv_conic[i])
      out_p[i] = p
           

class ConicPdf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, uv, uv_conic):
        p = torch.zeros_like(xy[:, 0])
        ctx.save_for_backward(xy, uv, uv_conic, p)
        kernel_conic_pdf(xy, uv, uv_conic, p)
        return p

    @staticmethod
    def backward(ctx, grad_p):
        xy, uv, uv_conic, _ = ctx.saved_tensors
        grad_uv = torch.zeros_like(uv)
        grad_conic = torch.zeros_like(uv_conic)
        kernel_conic_pdf_grad(xy, uv, uv_conic, grad_uv, grad_conic)

        grad_p = grad_p.unsqueeze(1)

        return None, grad_uv * grad_p, grad_conic * grad_p


def random_inputs(seed, n, device='cpu'):
    torch.random.manual_seed(seed)

    dx = torch.randn(n, 2, device=device, dtype=torch.float32)
    uv = torch.rand(n, 2, device=device, dtype=torch.float32) * 100

    conic = torch.randn(n, 3, device=device, dtype=torch.float32)
    return (uv + dx), uv, conic.exp()


def test_conic_grad( device='cpu'):
    xy, uv, uv_conic = random_inputs(0, 1000, device=device)

    out1, grads1 = eval_with_grad(torch_conic_pdf, xy, uv, uv_conic)
    out2, grads2 = eval_with_grad(ConicPdf.apply, xy, uv, uv_conic)

    assert torch.allclose(out1, out2, atol=1e-5), "out1 != out2"
    for grad1, grad2 in zip(grads1, grads2):
        if grad1 is None or grad2 is None:
            continue
        
        assert torch.allclose(grad1, grad2, atol=1e-5), "grad1 != grad2"


if __name__ == '__main__':
  test_conic_grad()
