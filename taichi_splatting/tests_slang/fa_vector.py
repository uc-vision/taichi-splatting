import numpy as np
import torch
import slangpy as spy
import os

from pathlib import Path

if __name__ == "__main__":

    betas=(0.9, 0.999)
    eps=1e-16
    bias_correction=True

    device = spy.create_device(type=spy.DeviceType.cuda)

    script_dir = "/local/cpe76/slang-workspace/taichi-splatting/taichi_splatting/optim_slang"
    shader_file = os.path.join(script_dir, "fractional_adam.slang")

    module = spy.Module.load_from_file(device, shader_file)

    bias_correction_int = int(bias_correction)

    N = 256
    M = 64 
    D = 16
    lr = 0.001

    lr_step = torch.rand((M, D), dtype=torch.float32, device='cuda')
    indexes = torch.randperm(N, device='cuda')[:M].to(torch.int)
    weight = torch.rand(M, dtype=torch.float32, device='cuda')

    m_arr = torch.rand((N, D), dtype=torch.float32, device='cuda')
    v_arr = torch.rand((N), dtype=torch.float32, device='cuda')

    total_weight = torch.rand(N, dtype=torch.float32, device='cuda')
    grad = torch.rand((N, D), dtype=torch.float32, device='cuda')


    slang_kernel = module.require_function(f'vector_kernel<{D}>')

    indexes_buff = spy.NDBuffer(shape=indexes.shape, dtype=module.int, device=device)
    indexes_buff.copy_from_torch(indexes)

    slang_kernel(
        lr_step=lr_step,
        index=indexes_buff,
        weight=weight,

        m_arr=m_arr,
        v_arr=v_arr,
        total_weight=total_weight,
        grad=grad,

        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,

        bias_correction=bias_correction_int,
    )