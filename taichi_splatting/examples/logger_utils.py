import time
import pandas as pd
import matplotlib.pyplot as plt


class TrainingLogger:
    def __init__(self):
        self.logs = []
        self.start_time = time.time()

    def log(self, iteration, psnr, n_points):
        elapsed_time = time.time() - self.start_time
        self.logs.append({
            'iteration': iteration,
            'time': elapsed_time,
            'psnr': psnr,
            'n': n_points
        })

    def save_csv(self, filename="training_log.csv"):
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"Saved training log to {filename}")

    def plot(self):
        df = pd.DataFrame(self.logs)

        # PSNR vs. Iteration
        plt.figure()
        plt.plot(df['iteration'], df['psnr'], label='PSNR')
        plt.xlabel('Iteration')
        plt.ylabel('PSNR')
        plt.title('Rendering Accuracy Over Iterations')
        plt.grid(True)
        plt.show()

        # PSNR vs. Time
        plt.figure()
        plt.plot(df['time'], df['psnr'], label='PSNR over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('PSNR')
        plt.title('Rendering Accuracy Over Time')
        plt.grid(True)
        plt.show()