from pathlib import Path
from matplotlib import pyplot
import numpy as np
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

def nonzero_mean(row):
    nonzero = [x for x in row if x > 0 and np.isfinite(x)]
    # return np.mean(nonzero) 
    return np.prod(nonzero) ** (1.0 / len(nonzero))

def plot_overall(df, name, axis_label="iters/second", backward=False):
    df = df.query(f"backward == {backward}").reset_index()

    g = sns.catplot(
      data=df, kind="bar",
      x="image_size", y="geometric_mean", hue="impl",
      palette="dark", alpha=.6, height=6, aspect=6/6
    )

    g.despine(left=True)
    g.set_axis_labels("image size (longest side)", axis_label)
    g.legend.set_title("")

    g.fig.subplots_adjust(top=0.9, bottom=0.2)
    g.fig.suptitle(f"{name} ({'forward+backward' if backward else 'forward'})", fontsize=16)
    g.set_xticklabels(rotation=45)
    


def plot_benchmarks(df, name, axis_label="iters/second", image_size=1024, backward=False):

    df = df.query(f"backward == {backward} and image_size == {image_size}"
                  ).reset_index().drop(columns=["image_size", "backward"])
    

    df = df.melt(id_vars=['impl'])

    g = sns.catplot(
      data=df, kind="bar",
      x="variable", y="value", hue="impl",
      palette="dark", alpha=.6, height=6, aspect=14/6
    )

    g.despine(left=True)
    g.set_axis_labels("image size (longest side)", axis_label)
    g.legend.set_title("")

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Benchmark {name} {'foward+backward' if backward else 'forward'} @ {image_size}", fontsize=16)
    g.set_xticklabels(rotation=30)

    
def make_speedup(df, ref_impl="gaussian-splatting"):
    speedups = []
    ref_impl = df.query(f"impl == '{ref_impl}'").droplevel(0)


    for impl in df.index.get_level_values('impl').unique():
      impl_df = df.query(f"impl == '{impl}'").droplevel(0)

      speedup = impl_df / ref_impl
      # speedup.dropna(inplace=True)

      old_idx = speedup.index.to_frame()
      old_idx.insert(0, 'impl', impl)
      speedup.index = pd.MultiIndex.from_frame(old_idx)

      speedups.append(speedup)

    return pd.concat(speedups)


def load_preprocess(filename):
  df = pd.read_csv(filename, index_col=[0, 1, 2], header=0)

  df.index = df.index.remove_unused_levels().set_levels(
      ["gaussian-splatting", "taichi-splatting(16)", "taichi-splatting(32)",  "taichi(original)"],level=0)


  values = [nonzero_mean(row) for _, row in df.iterrows()]
  df.insert(0, 'geometric_mean', values)

  df = df.sort_index(level=["image_size", "backward"])
  speedups = make_speedup(df)

  return df, speedups

def main():
  runs = {
    '2070':'benchmark-2070.csv',
    '3090':'benchmark-3090.csv',
    '4090':'benchmark-4090.csv'} 
  
  pd.set_option("display.precision", 2)

  dfs = {}

  for name, filename in runs.items():
    df, speedups = load_preprocess(Path("raw") / filename)
    dfs[name] = df

    df.to_csv(filename, float_format='%.2f')
    print(df)

    folder = Path(f"charts-{name}")
    folder.mkdir(exist_ok=True, parents=True)

    for image_size in df.index.get_level_values('image_size').unique():
      plot_benchmarks(df, name, axis_label="iters/second", image_size=image_size, backward=True)
      pyplot.savefig(folder / f"all-{image_size}.png") 

    plot_overall(df, f"Benchmark {name}", axis_label="iters/second", backward=True)
    pyplot.savefig(folder / "overall.png") 

    plot_overall(speedups, f"Speedup {name}", axis_label="relative speedup", backward=True)
    pyplot.savefig(folder / "speedup.png") 

if __name__ == '__main__':
  main()