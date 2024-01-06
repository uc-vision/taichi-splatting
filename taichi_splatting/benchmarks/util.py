import time
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

def profiled_benchmark(name, f, iters=100, warmup=1):
  for _ in range(warmup):
    f()

  with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
      for _ in tqdm(range(iters), desc=f"{name}"):
        f()
      torch.cuda.synchronize()

  result = prof.key_averages().table(sort_by="self_cuda_time_total", 
                                      row_limit=25, max_name_column_width=70)
  print(result)


  

def timed_benchmark(name, f, iters=100, warmup=1):
  for _ in range(warmup):
    f()

  start = time.time()
  for _ in tqdm(range(iters), desc=f"{name}"):
    f()

  torch.cuda.synchronize()
  end = time.time()

  print(f'{name}  {iters} iterations: {end - start:.3f}s at {iters / (end - start):.1f} iters/sec')


def benchmarked(name, f, iters=100, warmup=1, profile: bool = False):
  if profile:
    profiled_benchmark(name, f, iters, warmup)
  else:
    timed_benchmark(name, f, iters, warmup)
