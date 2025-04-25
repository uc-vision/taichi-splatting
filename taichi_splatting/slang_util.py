from functools import cache
from sgl import SlangCompilerOptions
import slangpy as spy
import torch

from slangpy.backend import Device,  DeviceType
from importlib.resources import files

shader_path = files('taichi_splatting') / 'slang'
slangpy_path = files('slangpy') / 'slang'


enable_debug = False

@cache
def get_device(defines:dict[str, str]={}):
  global enable_debug

  options = SlangCompilerOptions()
  options.include_paths=[shader_path, slangpy_path],
  options.defines=defines


  return Device(
        type= DeviceType.automatic,
        compiler_options=options,
        enable_debug_layers=enable_debug,
        enable_cuda_interop=True
  )


@cache
def load_module(filename):
    device = get_device()
    return spy.TorchModule.load_from_file(device, filename)

def get_float_type(torch_dtype:torch.dtype):
  type_map = {
      torch.float32: 'float',
      torch.float64: 'double',
      torch.float16: 'half'
  }
  assert torch_dtype in type_map, f"Unsupported float dtype {torch_dtype}"
  return type_map[torch_dtype]
