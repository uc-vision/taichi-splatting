
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

import taichi as ti

class TaichiQueue():
  executor: ThreadPoolExecutor = None

  @classmethod
  def init(cls, *args, **kwargs) -> None:
    if cls.executor is None:
      cls.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taichi",
        initializer=partial(ti.init, *args, **kwargs))
      
    return cls.executor
    
  @classmethod
  def queue(cls) -> ThreadPoolExecutor:
    assert cls.executor is not None, "TaichiQueue not initialized (run TaichiQueue.init() first)"
    return cls.executor
  
  @staticmethod
  def _await_run(func, *args) -> any:
    args = [arg.result() if isinstance(arg, Future) else arg for arg in args]
    return func(*args)
      
  @staticmethod
  def run_async(func, *args) -> Future:
    return TaichiQueue.queue().submit(TaichiQueue._await_run, func, *args)
  
  @staticmethod
  def run_sync(func, *args) -> any:
    return TaichiQueue.run_async(func, *args).result()
  
  @classmethod
  def stop(cls) -> None:
    executor = TaichiQueue.executor
    if executor is not None:
      cls.run_sync(ti.reset)
      executor.shutdown(wait=True)
      TaichiQueue.executor = None


def queued(kernel):
  def f(*args, **kwargs):
    return TaichiQueue.run_sync(kernel, *args, **kwargs)
  return f
