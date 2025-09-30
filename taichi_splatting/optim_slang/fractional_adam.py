#fractional_adam.py
import numpy as np
import torch
import slangpy as spy
import os

from pathlib import Path

def slang_scalar_kernel(dims, betas=(0.9, 0.999), eps=1e-16, bias_correction=True):

    device = spy.create_torch_device(type=spy.DeviceType.cuda, torch_device=torch.device('cuda', 0))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    shader_file = os.path.join(script_dir, "fractional_adam.slang")

    module = spy.Module.load_from_file(device, shader_file)

    bias_correction_int = int(bias_correction)
    slang_kernel = module.require_function(f'scalar_kernel<{dims}>')


    def kernel(
        lr_step: torch.Tensor,  # M, D
        indexes: torch.Tensor,  # M
        weight: torch.Tensor,  # M
        m_arr: torch.Tensor,  # N, D
        v_arr: torch.Tensor,  # N, D
        total_weight: torch.Tensor,  # N
        grad: torch.Tensor,  # N, 
        lr: float):
        

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

    return kernel

def slang_vector_kernel(betas=(0.9, 0.999), eps=1e-16, dims=3, bias_correction=True):
    device = spy.create_torch_device(type=spy.DeviceType.cuda, torch_device=torch.device('cuda', 0))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    shader_file = os.path.join(script_dir, "fractional_adam.slang")

    module = spy.Module.load_from_file(device, shader_file)

    bias_correction_int = int(bias_correction)
    slang_kernel = module.require_function(f'vector_kernel<{dims}>')

    def kernel(
        lr_step: torch.Tensor,  # M, D
        indexes: torch.Tensor,  # M
        weight: torch.Tensor,  # M
        m_arr: torch.Tensor,  # N, D
        v_arr: torch.Tensor,  # N, D
        total_weight: torch.Tensor,  # N
        grad: torch.Tensor,  # N, 
        lr: float):
        

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

    return kernel
    