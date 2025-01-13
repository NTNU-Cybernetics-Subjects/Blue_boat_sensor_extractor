import pandas as pd
import numpy as np
import fix_dataset
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.integrate import simpson, cumulative_trapezoid

from matlab_settings import matlab_settings, look_like_matlab

# plt.rc('text', usetex=True)

RAW_DATA_FOLDER = Path(__file__).resolve().parent.joinpath("raw_data")
raw_data_files = [file.name for file in RAW_DATA_FOLDER.glob("*.csv")]


def show_plot(block=False):
    plt.legend()
    # plt.grid(True)
    if block:
        plt.show()
        return

    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

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

def analyse_acc_bias_correction(df: pd.DataFrame):
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

    plot_acc(tt, u_dot, u_dot_corr, v_dot, v_dot_corr)

def analyse_acc_bias_correction_all_files(folder: Path, files: list):
    for file in files:
        print(f"Analysing file {file}")
        full_file = folder.joinpath(file)
        df = pd.read_csv(full_file)
        analyse_acc_bias_correction(df)

def plot_all_positions():
    for file in raw_data_files:
        print(file)
        raw_df = pd.read_csv(RAW_DATA_FOLDER.joinpath(file))
        df = fix_dataset.fix_dataset(raw_df)

        x = df['x_ned']
        y = df['y_ned']

        lat = df['lat']
        lon = df['lon']

        _, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 5))

        ax1.plot(y, x, label="ned")
        ax1.set_title("ned")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(lon, lat, label="gps")
        ax2.set_title("gps")
        ax2.grid(True)
        ax2.legend()

        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()

def fake_full():
    filename = "full_dataset.csv"
    df = pd.read_csv(filename)

    fake_df = pd.DataFrame({

        'timestamp(ms)':  df['original_timestamp'].to_numpy(),
        'GPS.Lat': df['lat'].to_numpy() / 10000000,
        'GPS.Lng': df['lon'].to_numpy() / 10000000,
        'GPS.Alt': df['alt'].to_numpy(),
        'DCM.Yaw': df['yaw'].to_numpy(),
        'IMU.GyrZ': df['yaw_rate'].to_numpy(),
        'IMU.AccX': df['surge_dot'].to_numpy(),
        'IMU.AccY': df['sway_dot'].to_numpy(),
        'RCOU.C1': df['thr_left'].to_numpy(),
        'RCOU.C3': df['thr_right'].to_numpy(),

    })
    return fake_df

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5*fs
    normal_cuttoff = cutoff/nyquist
    b, a = butter(order, normal_cuttoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def full_analyse_test(df: pd.DataFrame, ax:Axes):

    # Extract
    original_timestamp = df['original_timestamp'].to_numpy()

    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()
    alt = df['alt'].to_numpy()

    x_ned = df['x_ned'].to_numpy()
    y_ned = df['y_ned'].to_numpy()
    yaw = df['yaw'].to_numpy()

    surge = df['surge'].to_numpy()
    sway = df['sway'].to_numpy()
    yaw_rate = df['yaw_rate'].to_numpy()

    surge_dot = df['surge_dot'].to_numpy()
    surge_dot_bias = df['surge_dot_bias'].to_numpy()
    sway_dot = df['sway_dot'].to_numpy()
    sway_dot_bias = df ['sway_dot_bias'].to_numpy()
    yaw_acc = df['yaw_acc'].to_numpy()

    surge_dot_bias = df['surge_dot_bias'].to_numpy()
    sway_dot_bias = df['sway_dot_bias'].to_numpy()

    # thr_left = df['thr_left'].to_numpy()
    # thr_right = df['thr_right'].to_numpy()
    pwm_rotation = df['pwm_rotation'].to_numpy()
    pwm_forward = df['pwm_forward'].to_numpy()

    rotation_force = df["rotation_force"].to_numpy()
    forward_force = df["forward_force"].to_numpy()

    # right_force = df['right_force'].to_numpy()
    # left_force = df['left_force'].to_numpy()

    gps_y_dot_dot = df["gps_y_dot_dot"].to_numpy()
    gps_x_dot_dot = df["gps_x_dot_dot"].to_numpy()


    datapoints = len(df)
    step = 0.1
    end_time = round(datapoints*step, 1)
    t = np.arange(0, end_time, step)

    ## init labels

    ## Analyse
    surge_dot_savgol11 = savgol_filter(surge_dot, window_length=5, polyorder=3)
    surge_dot_savgol12 = savgol_filter(surge_dot, window_length=11, polyorder=3)
    surge_dot_savgol13 = savgol_filter(surge_dot, window_length=16, polyorder=3)
    surge_dot_savgol14 = savgol_filter(surge_dot, window_length=21, polyorder=3)

    surge_dot_savgol21 = savgol_filter(surge_dot, window_length=21, polyorder=1)
    surge_dot_savgol22 = savgol_filter(surge_dot, window_length=21, polyorder=3)
    surge_dot_savgol23 = savgol_filter(surge_dot, window_length=21, polyorder=6)
    surge_dot_savgol24 = savgol_filter(surge_dot, window_length=21, polyorder=8)

    surge_dot_butter11 = butter_lowpass_filter(surge_dot, 0.1, 1/step, order=5)
    surge_dot_butter12 = butter_lowpass_filter(surge_dot, 0.5, 1/step, order=5)
    surge_dot_butter13 = butter_lowpass_filter(surge_dot, 1, 1/step, order=5)
    surge_dot_butter14 = butter_lowpass_filter(surge_dot, 2, 1/step, order=5)

    surge_dot_butter21 = butter_lowpass_filter(surge_dot, 0.5, 1/step, order=1)
    surge_dot_butter22 = butter_lowpass_filter(surge_dot, 0.5, 1/step, order=3)
    surge_dot_butter24 = butter_lowpass_filter(surge_dot, 0.5, 1/step, order=6)
    surge_dot_butter23 = butter_lowpass_filter(surge_dot, 0.5, 1/step, order=8)

    surge_dot_savgol = savgol_filter(surge_dot, window_length=21, polyorder=3)
    surge_dot_butter = butter_lowpass_filter(surge_dot, 0.5, 1/step, order=5)

    surge_dot_trap = np.zeros(datapoints)
    surge_dot_trap[1:] = cumulative_trapezoid(surge_dot_butter,t, dx=step)

    surge_dot_integrated = fix_dataset.integrate(surge_dot_savgol, 0.1)

    sway_dot_filtered = savgol_filter(sway_dot, window_length=21, polyorder=3)
    sway_dot_integrated = fix_dataset.integrate(sway_dot_filtered, 0.1)

    alpha = 0.95
    surge_fuse = alpha * surge + (1-alpha)*surge_dot_trap
    sway_fuse = alpha * sway + (1-alpha)*sway_dot_integrated

    abs_speed = np.sqrt(surge**2 + sway**2)
    max_speed = np.full_like(t, 3.0)

    yaw_rate_savgol = savgol_filter(yaw_rate, window_length=21, polyorder=2)
    yaw_rate_butter = butter_lowpass_filter(yaw_rate, 0.5, 1/step)

    yaw_acc = fix_dataset.differentiate(yaw_rate_butter, step)
    yaw_acc_savgol = savgol_filter(yaw_acc, window_length=21, polyorder=2)
    yaw_acc_savgol = butter_lowpass_filter(yaw_acc, 0.5, 1/step)

    ## Plotting

    # ax.plot(t, surge_dot, label="Raw")

    # o = 3
    # ax.plot(t, surge_dot_savgol11, label="$w=5$")
    # ax.plot(t, surge_dot_savgol12, label="$w=11$")
    # ax.plot(t, surge_dot_savgol13, label="$w=16$")
    # ax.plot(t, surge_dot_savgol14, label="$w=21$")

    # w = 21
    # ax.plot(t, surge_dot_savgol11, label="$o=1$")
    # ax.plot(t, surge_dot_savgol12, label="$o=3$")
    # ax.plot(t, surge_dot_savgol13, label="$o=5$")
    # ax.plot(t, surge_dot_savgol14, label="$o=8$")

    # o = 5
    # ax.plot(t, surge_dot_butter11, label="$\\omega_c=0.1$")
    # ax.plot(t, surge_dot_butter12, label="$\\omega_c=0.5$")
    # ax.plot(t, surge_dot_butter13, label="$\\omega_c=6$")
    # ax.plot(t, surge_dot_butter14, label="$\\omega_c=8$")
    
    # omega_c = 0.5
    # ax.plot(t, surge_dot_butter21, label="$o=1$")
    # ax.plot(t, surge_dot_butter22, label="$o=3$")
    # ax.plot(t, surge_dot_butter23, label="$o=6$")
    # ax.plot(t, surge_dot_butter24, label="$o=8$")

    # ax.plot(t, surge_dot + surge_dot_bias, label="Raw")
    # ax.plot(t, surge_dot, label="Bias compensated")
    # ax.plot(t, surge_dot_savgol, label="Filtered from IMU")
    # ax.plot(t, surge_dot_butter, label="surge_dot_butter")
    # ax.plot(t, surge_dot_trap, label="surge trapz")
    # ax.plot(t, gps_x_dot_dot, label="Differentiated from GPS")

    # ax.plot(t, surge, label="surge (differentiate)")
    # ax.plot(t, surge_dot_integrated, label="surge (integrate)")
    # ax.plot(t, surge_fuse, label="surge fuse")

    # ax.plot(t, yaw_acc, label="yaw acc")
    # ax.plot(t, yaw_acc_savgol, label="yaw acc_filt")
    # ax.plot(t, yaw_rate, label="yaw rate")
    # ax.plot(t, yaw_rate_savgol, label="yaw rate savgol")
    # ax.plot(t, yaw_rate_butter, label="yaw rate butter")

    # ax.plot(t, sway_dot + sway_dot_bias[0], label="Raw")
    # ax.plot(t, sway_dot, label="Bias compensated")
    # ax.plot(t, sway_dot_filtered, label="Filtered")
    # ax.ylabel("acceleration in y [m/s^2]")
    #
    # ax.plot(t, sway_dot_integrated, label="sway (integrate)")
    # ax.plot(t, sway, label="sway (differentiate)")

    ax.plot(t, sway_dot_filtered, label="Filtered from IMU")
    ax.plot(t, gps_y_dot_dot, label="Differentiated from GPS")
    # ax.plot(t, sway_fuse, label="sway fuse")

    # ax.plot(t, right_force, label="right force")
    # ax.plot(t, left_force, label="left force")
    # ax.plot(t, pwm_forward / 100, label="pwm speed C3IN")
    # ax.plot(t, pwm_rotation / 100, label="pwm rot C1IN")

    # ax.plot(t, rotation_force, label="rotation force")
    # ax.plot(t, forward_force, label="forward force")

    # ax.plot(y_ned, x_ned)
    # ax.xlabel("y ned")
    # ax.ylabel("x ned")


    # ax.plot(t, sway, label="sawy")
    # ax.plot(t, abs_speed, label="abs_speed")
    # ax.plot(t, max_speed, label="max velcoity")
    
    # ax.set_title("Acceleration from IMU and derived from GPS", **matlab_settings["title"])
    ax.set_title("Acceleration in y-direction", **matlab_settings["title"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [$\\frac{m}{s}$]")
    # show_plot(True)

if __name__ == "__main__":
    # filename = "raw_data/00000232.csv"
    # filename = "raw_data/231.csv"
    filename = "raw_data/225.csv"
    # filename = "full_dataset.csv"
    # raw_df = fake_full()
    raw_df = pd.read_csv(filename)
    df = fix_dataset.fix_dataset(raw_df)

    # fig, ax = plt.subplots(figsize=(16,9))
    fig, ax = plt.subplots()
    
    look_like_matlab(ax)
    full_analyse_test(df, ax)

    ax.legend(**matlab_settings["legend"], loc="upper right")
    ax.legend(**matlab_settings["legend"])
    plt.show()

    fig.savefig("gps_acc_v_imu_sway.pdf", format="pdf")

    # plot_all_positions()
    #
    # step = 0.1
    # end_time = len(df)*step
    # t = np.arange(0, end_time, step)
    # surge_dot = # plot_acc(t, df['surge_dot'].to_numpy(), df['surge'].to_numpy(), df["sway_dot"].to_numpy(), df["sway"].to_numpy())
    #
    # plt.plot(df['x_ned'], df['y_ned'])
    # plt.grid()
    # fig.savefig("project_figures/compare_acceleration_IMU_GPS__surge.png")
    # plt.show()


