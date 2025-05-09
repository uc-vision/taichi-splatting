import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def plot_all_logs():
    csv_files = glob.glob("*.csv")
    plt.figure()
    for file in csv_files:
        df = pd.read_csv(file)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['iteration'], df['psnr'], label=label)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Iteration for All Logs')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    for file in csv_files:
        df = pd.read_csv(file)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['time'], df['psnr'], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Time for All Logs')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_all_logs()