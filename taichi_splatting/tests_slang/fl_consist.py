import torch
import numpy as np
import taichi as ti

from taichi_splatting.optim.fractional_laprop import scalar_kernel as taichi_scalar_kernel
from taichi_splatting.optim.fractional_laprop import vector_kernel as taichi_vector_kernel
from taichi_splatting.optim_slang.fractional_laprop import slang_scalar_kernel, slang_vector_kernel
from taichi_splatting import TaichiQueue

def random_inputs(n, m, d, vector=False, device='cuda'):

    if vector:
        v_arr = torch.rand(n, device=device)
    else:
        v_arr = torch.rand(n, d, device=device)

    return {
        "lr_step": torch.randn(m, d, device=device),
        "indexes": torch.from_numpy(
            np.random.choice(n, m, replace=False).astype(np.int64)
        ).to(device),
        "weight": torch.rand(m, device=device),
        "m_arr": torch.randn(n, d, device=device),
        "v_arr": v_arr,
        "total_weight": torch.rand(n, device=device),
        "grad": torch.randn(n, d, device=device),
        "lr": 0.1,
    }

def check_consistency(ref_kernel, test_kernel, n, m, d, vector=False, device='cuda'):
    inputs = random_inputs(n, m, d, vector, device)

    ref_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    test_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    test_inputs["indexes"] = test_inputs["indexes"].to(torch.int32)

    ref_kernel(**ref_inputs)
    test_kernel(**test_inputs)

    for name, ref_tensor in ref_inputs.items():
        if isinstance(ref_tensor, torch.FloatTensor):
            test_tensor = test_inputs[name]
            assert torch.allclose(ref_tensor, test_tensor, atol=1e-4), f"Tensor {name} is not consistent"

def test_fl_consistency():

    TaichiQueue.init(arch=ti.cuda, device_memory_GB=0.1, threaded=True)

    check_consistency(
        taichi_scalar_kernel(),
        slang_scalar_kernel(),
        n=100, m=50, d=3, vector=False,
    )
    print("Scalar Consistant!")

    check_consistency(
        taichi_vector_kernel(),
        slang_vector_kernel(),
        n=100, m=50, d=3, vector=True,
    )
    print("Vector Consistant!")

if __name__ == '__main__':
    test_fl_consistency()