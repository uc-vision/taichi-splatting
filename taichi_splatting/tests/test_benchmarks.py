from taichi_splatting.benchmarks import bench_projection, bench_rasterizer, bench_sh, bench_tilemapper

# currently takes a little too long to compile backward kernels
# def test_bench_projection():
#   args = bench_projection.parse_args([])
#   bench_projection.bench_projection(args)

def test_bench_rasterizer():
  args = bench_rasterizer.parse_args([])
  bench_rasterizer.bench_rasterizer(args)

def test_bench_sh():
  args = bench_sh.parse_args([])
  bench_sh.bench_sh(args)

def test_bench_tilemapper():
  args = bench_tilemapper.parse_args([])
  bench_tilemapper.bench_tilemapper(args)

def test_bench_projection():
  args = bench_projection.parse_args([])
  bench_projection.bench_projection(args)


def main():

  test_bench_tilemapper()
  test_bench_projection()

  test_bench_rasterizer()
  test_bench_sh()






if __name__ == '__main__':
  main()