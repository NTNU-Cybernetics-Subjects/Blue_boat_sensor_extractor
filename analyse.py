import pandas as pd
import numpy as np
import fix_dataset
from pathlib import Path

from matplotlib import pyplot as plt


RAW_DATA_FOLDER = Path(__file__).resolve().parent.joinpath("raw_data")
raw_data_files = [file.name for file in RAW_DATA_FOLDER.glob("*.csv")]
#
def analyse(df: pd.DataFrame):
    t = df['timestamp(ms)'].to_numpy()
    u_dot = df['IMU.AccX'].to_numpy()
    v_dot = df['IMU.AccY'].to_numpy()


    duration = (t[-1] - t[0])/1000 # secounds
    # print("duration", duration)
    step = 0.1
    tt = np.arange(0, duration+step, step)

    u_bias = fix_dataset.calculate_bias_acc(u_dot)
    u_dot_corr = fix_dataset.correct_for_bias(u_dot, u_bias)

    v_bias = fix_dataset.calculate_bias_acc(v_dot)
    v_dot_corr = fix_dataset.correct_for_bias(v_dot, v_bias)

    plot_acc(t, u_dot, u_dot_corr, v_dot, v_dot_corr)
#
def plot_acc(t, u_dot, u_dot_corr, v_dot, v_dot_corr):
    _, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 5))

    ax1.plot(t, u_dot, label="u_dot")
    ax1.plot(t, u_dot_corr, label="u_dot_corr")
    ax1.set_title("u_dot")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, v_dot, label="v_dot")
    ax2.plot(t, v_dot_corr, label="v_dot_corr")
    ax2.set_title("v_dot")
    ax2.grid(True)
    ax2.legend()

    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()
#
#
def analyse_all():
    for file in raw_data_files:
        print(file)
        full_file = RAW_DATA_FOLDER.joinpath(file)
        df = pd.read_csv(full_file)
        analyse(df)

if __name__ == "__main__":
    analyse_all()

# df = pd.read_csv("full_dataset.csv")
# step = 0.1
# end_time = len(df)*step
# t = np.arange(0, end_time, step)
# plot_acc(t, df['surge_dot'].to_numpy(), df['surge'].to_numpy(), df["sway_dot"].to_numpy(), df["sway"].to_numpy())





