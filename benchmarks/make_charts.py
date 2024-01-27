from matplotlib import pyplot
import numpy as np
import seaborn as sns
import pandas as pd


def nonzero_mean(row):
    nonzero = [x for x in row if x > 0]
    # return np.mean(nonzero) 
    return np.prod(nonzero) ** (1.0 / len(nonzero))

def plot_overall(df, name, axis_label="iters/second ($s^{-1}$)", backward=False):
    df = df.query(f"backward == {backward}").reset_index()

    g = sns.catplot(
      data=df, kind="bar",
      x="image_size", y="geometric mean", hue="impl",
      palette="dark", alpha=.6, height=6
    )

    g.despine(left=True)
    g.set_axis_labels("image size (longest side)", axis_label)
    g.legend.set_title("")

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Benchmark {name} ({'forward+backward' if backward else 'forward'})", fontsize=16)
    
    pyplot.show()

def plot_benchmarks(df, name, axis_label="iters/second ($s^{-1}$)", image_size=1024, backward=False):

  
    df = df.query(f"backward == {backward} and image_size == {image_size}"
                  ).reset_index().drop(columns=["image_size", "backward"])
    

    df = df.melt(id_vars=['impl'])

    g = sns.catplot(
      data=df, kind="bar",
      x="variable", y="value", hue="impl",
      palette="dark", alpha=.6, height=6
    )

    g.despine(left=True)
    g.set_axis_labels("image size (longest side)", axis_label)
    g.legend.set_title("")

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Benchmark {name} {'foward+backward' if backward else 'forward'} @ {image_size}", fontsize=16)
    
    pyplot.show()



def make_speedup(df, ref_impl="gaussian-splatting"):
    speedups = []
    ref_impl = df.query(f"impl == '{ref_impl}'").droplevel(0)


    for impl in df.index.get_level_values('impl').unique():
      impl_df = df.query(f"impl == '{impl}'").droplevel(0)

      speedup = impl_df / ref_impl
      speedup.dropna(inplace=True)

      old_idx = speedup.index.to_frame()
      old_idx.insert(0, 'impl', impl)
      speedup.index = pd.MultiIndex.from_frame(old_idx)

      speedups.append(speedup)

    return pd.concat(speedups)



def main():
  runs = {
    '3090':'benchmark-3090.csv',
    '4090':'benchmark-4090.csv'} 
  
  pd.set_option("display.precision", 2)

  for name, file in runs.items():
    df = pd.read_csv(file, index_col=[0, 1, 2], header=0)

    df.index = df.index.remove_unused_levels().set_levels(
       ["gaussian-splatting", "taichi-splatting", "taichi-splatting(32)",  "taichi(original)"],level=0)

    print(df)

    # df = df.query("impl != 'taichi-splatting(32)'")

    print(f"Benchmark: {name}")

    values = [nonzero_mean(row) for _, row in df.iterrows()]
    df.insert(0, 'geometric mean', values)

    df = df.sort_index()
    speedups = make_speedup(df)

    print(speedups.query("backward == False"))
    print(speedups.query("backward == True"))

    plot_benchmarks(df, name, axis_label="iters/second", image_size=4096, backward=True)
    
    plot_overall(df, name, axis_label="iters/second", backward=True)
    plot_overall(speedups, name, axis_label="iters/second", backward=True)



if __name__ == '__main__':
  main()