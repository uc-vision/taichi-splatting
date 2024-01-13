from beartype import beartype
import taichi as ti
import torch

from taichi_splatting.misc.autograd import restore_grad
from taichi_splatting.taichi_lib.conversions import torch_taichi


def indexing_function(size:int, dtype=torch.float32):   

  ti_type = torch_taichi[dtype]
  vec = ti.types.vector(n=size, dtype=ti_type)

  @ti.kernel
  def indexing_kernel(
    features: ti.types.ndarray(dtype=vec, ndim=1),  # (N, C)
    indexes: ti.types.ndarray(dtype=ti.i64, ndim=1),  # (M)
    output: ti.types.ndarray(dtype=vec, ndim=1),  # (M, C)
  ):
    for i in range(output.shape[0]):
      output[i] = features[indexes[i]]

  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indexes):
      device = features.device

      n = indexes.shape[0]
      features_out = torch.empty((n, features.shape[1]), dtype=features.dtype, device=device)
      indexing_kernel(features, indexes, features_out)

      ctx.indexes = indexes

      ctx.save_for_backward(features, features_out)
      return features_out

    @staticmethod
    def backward(ctx, dfeatures_out):
      features, features_out = ctx.saved_tensors

      with restore_grad(features, features_out):
        features_out.grad = dfeatures_out.contiguous()
        indexing_kernel.grad(
          features, ctx.indexes, features_out)

        return features.grad, None

  return _module_function


@beartype
def index_features(features:torch.Tensor, indexes:torch.Tensor):
  d = features.shape[1:]
  features = features.view(features.shape[0], -1)

  _module_function = indexing_function(features.shape[1], features.dtype)
  return _module_function.apply(features, indexes.contiguous()).view(-1, *d)


