import copy
from functools import cached_property
from typing import Callable
import torch.optim as optim
import torch



class ParameterClass():
  """
  Maintains a group of mixed parameters/non parameters in a TensorClass,
  as well as optimizer state synchronized with the parameters.

  Note - call ParameterClass.create to create a ParameterClass instead 
  of the constructor directly.

  Supports filtering with python indexing and appending parameters.

  Parameters:
    tensors: TensorClass
    param_groups: list of parameter dicts, see torch.optim for details 
    param_state: dict of optimizer state to insert into the optimizer
  """

  def __init__(self, tensors, param_groups, param_state=None):
    self.tensors = tensors
    self.optimizer = optim.Adam(param_groups, fused=True)

    if param_state is not None:
      for k, v in param_state.items():
        self.optimizer.state[self.tensors[k]] = v


  @staticmethod
  def create(tensors, learning_rates, base_lr=1.0):
    param_dict = as_parameters(tensors, learning_rates.keys())
    param_groups = [
      dict(params=[param_dict[name]], lr=lr * base_lr, name=name)
        for name, lr in learning_rates.items()
    ]

    return ParameterClass(tensors, param_groups)


  def zero_grad(self):
    self.optimizer.zero_grad()


  def step(self):
    self.optimizer.step()


  def __getattr__(self, name):
    return getattr(self.tensors, name)

  @cached_property
  def _optimized_keys(self):
    return {group["name"] for group in self.optimizer.param_groups}
    
  def _updated_state(self, f:Callable, state_keys = ("exp_avg", "exp_avg_sq")):

    def modify_state(state):
      return {k : f(v) if k in state_keys else state[k]
                for k, v in state.items()}

    return {k:modify_state(self.optimizer.state[param])
              for k, param in self.tensors.items()
                if param in self.optimizer.state}
  

  def _updated_parameters(self, tensors):
    tensors = as_parameters(tensors, self.optimized_keys)
    updated_groups = [ replace_dict(group, params=[tensors[group["name"]]])
      for group in self.optimizer.param_groups]
    
    return tensors, updated_groups

  def __index__(self, idx):
    tensors, updated_groups = self.updated_parameters(self.tensors[idx])
    state = self._updated_state(lambda x: x[idx])
    return ParameterClass(tensors, updated_groups, state)
  
  def append(self, tensors):
    n = tensors.batch_size[0]

    tensors, updated_groups = self._updated_parameters(torch.cat([self.tensors, tensors]))
    state = self._updated_state(lambda x: torch.cat(
      [x, x.new_zeros(n)])
    )
    
    return ParameterClass(tensors, updated_groups, state)



def as_parameters(tensors, keys):
    param_dict = {k: torch.nn.Parameter(x, requires_grad=True) 
                        for k, x in tensors.items()
                        if k in keys}

    cls = type(tensors)
    return cls.from_dict(param_dict) 


def replace_dict(d, **kwargs):
  d = copy(d)
  d.update(kwargs)
  return d