import torch
import numpy as np
import taichi as ti

from taichi_splatting.optim.fractional_laprop import scalar_kernel as taichi_scalar_kernel
from taichi_splatting.optim.fractional_laprop import vector_kernel as taichi_vector_kernel
from taichi_splatting.optim_slang.fractional_laprop import slang_scalar_kernel, slang_vector_kernel
from taichi_splatting import TaichiQueue

from functools import partial

from taichi_splatting.benchmarks.util import benchmarked

def random_inputs(n, m, d, vector=False, device='cuda'):
    if vector:
        v_arr = torch.rand(n, device=device)
    else:
        v_arr = torch.rand(n, d, device=device)

    return {
        "lr_step": torch.empty(m, d, device=device),
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

def run_benchmark(name, kernel, inputs, is_slang):
    bench_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    if is_slang:
        bench_inputs["indexes"] = bench_inputs["indexes"].to(torch.int32)
    else:
        bench_inputs["indexes"] = bench_inputs["indexes"].to(torch.int64)

    test_fn = partial(kernel, **bench_inputs)
    benchmarked(name, test_fn, profile=False, iters=10000, warmup=100)


def bench_fl():
    TaichiQueue.init(arch=ti.cuda, device_memory_GB=0.1, threaded=True)

    n, m, d = 100000, 50000, 16

    print(f"Benchmarking Fractional Laprop n={n}, m={m}, d={d}")

    scalar_inputs = random_inputs(n, m, d, vector=False)
    vector_inputs = random_inputs(n, m, d, vector=True)

    # Scalar kernels
    print("\n--- Scalar kernels ---")
    run_benchmark("Taichi scalar", taichi_scalar_kernel(), scalar_inputs, is_slang=False)
    run_benchmark("Slang scalar", slang_scalar_kernel(), scalar_inputs, is_slang=True)

    # Vector kernels
    print("\n--- Vector kernels ---")
    run_benchmark("Taichi vector", taichi_vector_kernel(dims=d), vector_inputs, is_slang=False)
    run_benchmark("Slang vector", slang_vector_kernel(dims=d), vector_inputs, is_slang=True)


if __name__ == '__main__':
    bench_fl()