from functools import cache
import torch
import math

from .slang_util import get_float_type, load_module


@cache 
def get_sh_functions(dtype:torch.dtype):
    type_str = get_float_type(dtype)

    module = load_module("spherical_harmonics.slang")
    functions = {
        0: module.require_function(f'evaluate_sh0<{type_str}>'),
        1: module.require_function(f'evaluate_sh1<{type_str}>'),
        2: module.require_function(f'evaluate_sh2<{type_str}>'),
        3: module.require_function(f'evaluate_sh3<{type_str}>')
    }

    return functions

def check_sh_degree(sh_features):
    assert len(sh_features.shape) == 3, f"SH features must have 3 dimensions, got {sh_features.shape}"
    n_sh = sh_features.shape[2]
    n = int(math.sqrt(n_sh))
    assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
    return (n - 1)

def evaluate_sh_at(sh_params: torch.Tensor,   # M, K (degree + 1)^2, (usually K=3, for RGB)
                  positions: torch.Tensor,    # M, 3 (packed gaussian or xyz)
                  indexes: torch.Tensor,      # N, 1 (indexes to gaussians) 0 to M 
                  camera_pos: torch.Tensor    # 3
                  ) -> torch.Tensor:          # N, K
    
    degree = check_sh_degree(sh_params)
    func = get_sh_functions(sh_params.dtype)[degree]
    

    if indexes.shape[0] == 0:
      return torch.zeros_like(positions, requires_grad=sh_params.requires_grad)
    
    result = func (
        sh_params=sh_params[indexes],
        position=positions[indexes], 
        camera_pos=camera_pos
    )

    return result