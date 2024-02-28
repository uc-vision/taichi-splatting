

from beartype.typing import Sequence
import torch
from tqdm import tqdm




def eval_with_grad(f, *args):
  args = [x.detach().clone().requires_grad_(x.requires_grad) for x in args]

  # multiply by one to make the argument not a leaf tensor 
  # avoids an issue with taichi/pytorch autograd integration
  out = f(*args)
  loss = 0

  if isinstance(out, tuple):
    for o in out:
      loss += o.mean()
  else:
    loss = out.mean()

  loss.backward()

  return out, tuple(arg.grad for arg in args)

def allclose(test_name, name, a, b, atol=1e-2, rtol=1e-3):
  assert type(a) == type(b), f"{name}: type does not match {type(a)} != {type(b)}"
  if isinstance(a, torch.Tensor):
    if not torch.allclose(a, b, atol=atol):
      print(a)
      print(b)
      
      raise AssertionError(f"{test_name}.{name} does not match")

  elif isinstance(a, Sequence):
    for a_, b_, name_ in zip(a, b, name):
      allclose(test_name, name_, a_, b_, atol=atol, rtol=rtol)


def compare_with_grad(test_name, input_names, output_names,
      f1, f2, gen_inputs, iters=100):

  seeds = torch.randint(0, 10000, (iters, ))
  for seed in tqdm(seeds, desc=test_name):
      with torch.no_grad():
        inputs  = gen_inputs(seed)

      out1, grads1 = eval_with_grad(f1, *inputs)
      out2, grads2 = eval_with_grad(f2, *inputs)


      try:
        allclose(test_name, output_names, out1, out2, atol=1e-5)
        allclose(f"{test_name}_grad", input_names, grads1, grads2, atol=1e-5)
      except AssertionError as e:
        print(f"Failed at seed {seed}")
        raise e
