
import warnings
# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()
warnings.filterwarnings('ignore') 


def eval_with_grad(f, *args):
  args = [args.detach().clone().requires_grad_(True) for args in args]

  # multiply by one to make the argument not a leaf tensor 
  # avoids an issue with taichi/pytorch autograd integration
  out = f(*[arg * 1.0 for arg in args])
  out.sum().backward()
  return out, [arg.grad for arg in args]