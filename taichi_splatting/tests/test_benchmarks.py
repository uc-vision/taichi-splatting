from taichi_splatting.benchmarks import bench_projection, bench_rasterizer, bench_sh

def test_bench_projection():
  bench_projection.test_projection()

def test_bench_rasterizer():
  bench_rasterizer.test_rasterizer()

def test_bench_sh():
  bench_sh.test_sh()