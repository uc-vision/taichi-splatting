from contextlib import contextmanager

import torch

@contextmanager
def restore_grad(*tensors):
  try:
      grads = [tensor.grad if tensor.grad is not None else None
                for tensor in tensors]
      
      for tensor in tensors:    
          if tensor.requires_grad is True:
            tensor.grad = torch.zeros_like(tensor)
      yield
  finally:
      for tensor, grad in zip(tensors, grads):
          tensor.grad = grad