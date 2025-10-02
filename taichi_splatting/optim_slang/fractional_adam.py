#fractional_adam.py
import numpy as np
import torch
import slangpy as spy
import os

from pathlib import Path

def create_cuda_device():
    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()
    with torch.device('cuda:0'):
        handles = spy.get_cuda_current_context_native_handles()


    shaderpath = str(Path(spy.__file__).parent / "slang")

    options = spy.SlangCompilerOptions()
    options.optimization = spy.SlangOptimizationLevel.maximal
    options.include_paths = [shaderpath]

    return spy.Device(
        type=spy.DeviceType.cuda,
        compiler_options=options,
        existing_device_handles=handles
    )

def slang_scalar_kernel(dims, betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    device = create_cuda_device()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    shader_file = os.path.join(script_dir, "fractional_adam.slang")
    
    module = spy.Module.load_from_file(device, shader_file)

    slang_kernel = module.require_function(f'scalar_kernel<{dims}, {str(bias_correction).lower()}>')


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
            eps=eps
        )

    return kernel

def slang_vector_kernel(betas=(0.9, 0.999), eps=1e-16, dims=3, bias_correction=True):
    device = create_cuda_device()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    shader_file = os.path.join(script_dir, "fractional_adam.slang")
    
    module = spy.Module.load_from_file(device, shader_file)

    slang_kernel = module.require_function(f'vector_kernel<{dims}, {str(bias_correction).lower()}>')

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
            eps=eps
        )

    return kernel
    