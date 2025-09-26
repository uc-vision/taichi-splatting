from functools import cache
import slangpy as spy
import torch

from slangpy import DeviceType
# from slangpy.core.utils import create_torch_device
from importlib.resources import files

from pathlib import Path


shader_path = Path(files('taichi_splatting')) / 'slang'
slangpy_path = Path(files('slangpy')) / 'slang'


@cache
def get_device(torch_device:torch.device, defines:dict[str, str]={}, debug:bool=False):


  # return spy.create_torch_device(
  #       type= DeviceType.cuda,
  #       include_paths=[str(shader_path), str(slangpy_path)],
  #       enable_debug_layers=debug,
  #       torch_device=torch_device
  # )

  return spy.create_device(
        type= DeviceType.cuda,
        include_paths=[str(shader_path), str(slangpy_path)],
        enable_debug_layers=debug,
        # torch_device=torch_device
  )


@cache
def load_module(filename, torch_device:torch.device):
    device = get_device(torch_device)
    return spy.Module.load_from_file(device, filename)

def get_float_type(torch_dtype:torch.dtype):
  type_map = {
      torch.float32: 'float',
      torch.float64: 'double',
      torch.float16: 'half'
  }
  assert torch_dtype in type_map, f"Unsupported float dtype {torch_dtype}"
  return type_map[torch_dtype]
