import copy
from beartype import beartype
from beartype.typing import Callable,  Dict, Optional
from tensordict import TensorDict
import torch.optim as optim
import torch
from functools import partial

default_optimizer = partial(optim.Adam, foreach=True, betas=(0.7, 0.999), amsgrad=True, weight_decay=0.0)

def replace_dict(d, **kwargs):
  d = copy.copy(d)
  d.update(kwargs)
  return d

class ParameterClass():
  """
  Maintains a group of mixed parameters/non parameters in a TensorClass,
  as well as optimizer state synchronized with the parameters.

  Note - call ParameterClass.create to create a ParameterClass instead 
  of the constructor directly.

  Supports filtering with python indexing and appending parameters.

  Parameters:
    tensors: TensorDict
    parameter_groups: dict of key -> parameter dict for parameters to be optimized, 
                      keys must exist in tensors dict
                      
    param_state: dict of key -> optimizer state to insert into the optimizer
  """

  def __init__(self, tensors:TensorDict, 
               parameter_groups:Dict[str, Dict], 
               param_state:Optional[Dict[str, torch.Tensor]]=None,
               optimizer = default_optimizer):

    self.tensors:TensorDict = as_parameters(tensors, parameter_groups.keys())


    param_groups = [
      dict(params=[self.tensors[name]], name=name, **group)
        for name, group in parameter_groups.items()
    ]

    self.create_optimizer = optimizer
    self.optimizer = self.create_optimizer(param_groups)
    

    if param_state is not None:
      for k, v in param_state.items():
        self.optimizer.state[self.tensors[k]] = v


  @beartype
  @staticmethod
  def create(tensors:TensorDict, parameter_groups:Dict[str, Dict], base_lr=1.0,
               optimizer = default_optimizer):
    
    parameter_groups = {k: replace_dict(group, lr=group.get("lr", 1.0) * base_lr)
          for k, group in parameter_groups.items()}
    
    return ParameterClass(tensors, parameter_groups, optimizer=optimizer)
  
  @property
  def parameter_groups(self):

    return {group['name']: {k: v for k, v in group.items() if k not in ['params', 'name']}
      for group in self.optimizer.param_groups}


  @property
  def learning_rates(self):
    return {group['name']: group['lr'] for group in self.optimizer.param_groups}


  @beartype
  def set_learning_rate(self, **kwargs:float):
    learning_rates = replace_dict(self.learning_rates, **kwargs)

    for group in self.optimizer.param_groups:
      group['lr'] = learning_rates[group['name']]

    return self

  def zero_grad(self):
    self.optimizer.zero_grad()


  def step(self, **kwargs):
    self.optimizer.step(**kwargs)


  def keys(self):
    return self.tensors.keys()

  def optimized_keys(self):
    return self.parameter_groups.keys()
    
  def items(self):
    return self.tensors.items()


  def __getattr__(self, name):
    if name  not in self.tensors.keys():
      raise AttributeError(f'ParameterClass has no attribute {name}')
    
    return self.tensors[name]


  def to(self, device):
    return ParameterClass(
      as_parameters(self.tensors.to(device), self.optimized_keys()), 
      self.parameter_groups, 
      self._updated_state(lambda x: x.to(device),
      optimizer=self.create_optimizer)
    )

  def to_dict(self):
    return self.tensors.to_dict()
  
  @property
  def batch_size(self):
    return self.tensors.batch_size


  def _updated_state(self, f:Callable):
    def modify_state(state):
      return {k : f(v) if torch.is_tensor(v) and v.dim() > 0 else state[k]
                for k, v in state.items()}

    return {name:modify_state(self.optimizer.state[param])
              for name, param in self.tensors.items()
                if param in self.optimizer.state}
    

  def get_state(self):
    return self._updated_state(lambda x: x)


  def __getitem__(self, idx):
    state = self._updated_state(lambda x: x[idx])
    return ParameterClass(self.tensors[idx], self.parameter_groups, state, optimizer=self.create_optimizer)
  
  def append_tensors(self, tensors:TensorDict):
    assert set(tensors.keys()) == set(self.tensors.keys()), f"{tensors.keys()} != {self.tensors.keys()}"
    n = tensors.batch_size[0]

    state = self._updated_state(lambda x: torch.cat(
      [x, x.new_zeros(n, *x.shape[1:])] )
    )
    return ParameterClass(torch.cat([self.tensors, tensors]), self.parameter_groups, state, optimizer=self.create_optimizer)

  def append(self, params:'ParameterClass'):
    return self.append_tensors(params.tensors)


def as_parameters(tensors, keys):
  
    param_dict = {
      k: torch.nn.Parameter(x.detach(), requires_grad=True) 
            if k in keys else x 
        for k, x in tensors.items()}

    cls = type(tensors)
    return cls.from_dict(param_dict) 





def concat_states(state, other_state):
  def concat_dict(d1, d2):

    return {k: torch.concat( (d1[k], d2[k]) ) if d1[k].dim() > 0 else d1[k]
            for k in d1.keys()}

  return {k: concat_dict(state[k], other_state[k]) 
          for k in state.keys()}