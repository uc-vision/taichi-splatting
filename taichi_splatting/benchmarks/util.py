import time
import taichi as ti
import torch
from tqdm import tqdm

def benchmarked(name, f, iters=100, warmup=1, profile=False):

  for _ in range(warmup):
    f()

  if profile:
    ti.profiler.clear_kernel_profiler_info()

  start = time.time()
  for _ in tqdm(range(iters), desc=f"{name}"):
    f()

  torch.cuda.synchronize()
  ti.sync()

  end = time.time()

  print(f'{name}  {iters} iterations: {end - start:.3f}s at {iters / (end - start):.1f} iters/sec')

  if  profile:
    ti.profiler.print_kernel_profiler_info("count")