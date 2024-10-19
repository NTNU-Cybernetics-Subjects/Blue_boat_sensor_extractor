from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from navpy import lla2ned
import navpy

RAW_DATA_FOLDER = Path(__file__).resolve().parent.joinpath("raw_data")
sorted_data_folder = Path(__file__).resolve().parent.joinpath("sorted_data")

raw_data_files = [file.name for file in RAW_DATA_FOLDER.glob("*.csv")]

# Basestation-bench  (found on google maps)
REF_LON = 10.314897
REF_LAT = 63.401783
REF_ATT = 0

def calculate_bias_acc(array):
    bin = np.arange(-1, 1, 0.01)

    hist, bin = np.histogram(array, bin)

    bias = (bin[np.argmax(hist)] + bin[np.argmax(hist)+1])/2 # Get acerage of that bin value

    return bias

def calculate_pwm_zero_point(array):
    bin = np.arange(1450, 1550, 2)

    hist, bin = np.histogram(array, bin)

    bias = (bin[np.argmax(hist)] + bin[np.argmax(hist)+1])/2 # Get acerage of that bin value

    return bias

def correct_for_bias(array, bias):
    return array - bias

def map_rcou_1_value(old_value, zero_point):

    if old_value == zero_point:
        return 0

    elif 1100 <= old_value < zero_point:
        # Linear increase from 1100 to 1505 (0 to 5.63)
        return 5.63 * (zero_point - old_value) / (zero_point - 1100)

    elif 1505 < old_value <= 1900:
        # Linear decrease from 1505 to 1900 (0 to -2.81)
        return -2.81 * (old_value - zero_point) / (1900 - zero_point)

    elif old_value <= 1100:
        # For values under 1100 just map it to 5.63
        return 5.63

    elif old_value >= 1900:
        # For values over 1900 just map it to -2.81
        return -2.81

# RCOU_C3 = right motor
def map_rcou_3_value(old_value, zero_point):

    if old_value == zero_point:
        return 0

    elif 1100 <= old_value < zero_point:
        # Linear decrease from 1505 to 1900 (0 to -2.81)
        return -2.81 * (zero_point - old_value) / (zero_point - 1100)

    elif 1505 < old_value <= 1900:
        # Linear increase from 1100 to 1505 (0 to 5.63)
        return 5.63 * (old_value - zero_point ) / (1900 - zero_point)

    elif old_value <= 1100:
        # For values under 1100 just map it to 5.63
        return -2.81

    elif old_value >= 1900:
        # For values over 1900 just map it to -2.81
        return 5.63

def integrate(array, dt):
    n = len(array)
    out = np.zeros(n)
    out[0] = array[0]*dt
    for i in range(1, n):
        # out[i] = out[i-1] + array[i-1]*dt
        out[i] = out[i-1] + 0.5 * (array[i] + array[i-1])*dt

    return out

def differentiate(array, dt):
    n = len(array)

    dx_dt = np.zeros(n)
    dx_dt[1:-1] = (array[2:] - array[:-2])/(2*dt)
    return dx_dt
    

def latlon_2_ned(lat, lon, att):
    ned = lla2ned(
        lat, lon, att,
        REF_LAT, REF_LON, REF_ATT
    )

    x_n = ned[:,0]
    y_n = ned[:,1]
    z_n = ned[:,2]

    return x_n, y_n, z_n

def ned_to_bodyfixed(xn,yn,yaw)-> tuple[np.dtype, np.dtype]:
    """Rotate ned frame to body-fixed
    returns: (xb, yb)"""
    R = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])
    vn = np.array([
        [xn],
        [yn]

    ])
    vb = np.dot(R, vn).flatten()
    return vb[0], vb[1]

def validate_timestamps(time_steps: np.ndarray) -> bool:

    step_ms = 100.0 # step size in ms
    # Check if the step lenght are constant in the dataset
    diff = np.diff(time_steps)
    check = np.where(diff != step_ms)[0]
    valid = len(check) <= 0
    assert(valid)
    return valid


def fix_all_data(df: pd.DataFrame):
    t = df['timestamp(ms)'].to_numpy()
    lat = df['GPS.Lat'].to_numpy() / 10000000
    lon = df['GPS.Lng'].to_numpy() / 10000000
    alt = df['GPS.Alt'].to_numpy()

    yaw = df['DCM.Yaw'].to_numpy()
    r = df['IMU.GyrZ'].to_numpy()

    u_dot = df['IMU.AccX'].to_numpy()
    v_dot = df['IMU.AccY'].to_numpy()

    thr_left = df['RCOU.C1'].to_numpy()
    thr_right = df['RCOU.C3'].to_numpy()

    xn, yn, zn = latlon_2_ned(lat, lon, alt)

    # Check that the timestamp are consistent in the dataset
    validate_timestamps(t)

    step = 0.1
    g = 9.81

    # correct acc signals
    u_bias = calculate_bias_acc(u_dot)
    v_bias = calculate_bias_acc(v_dot)

    u_dot_corr = correct_for_bias(u_dot, u_bias)
    v_dot_corr = correct_for_bias(v_dot, v_bias)

    # Map pwm thruster signals to force
    vectorized_left_thr_fun = np.vectorize(map_rcou_1_value)
    vectorized_right_thr_fun = np.vectorize(map_rcou_3_value)

    left_pwm_zero_point = calculate_pwm_zero_point(thr_left)
    right_pwm_zero_point = calculate_pwm_zero_point(thr_right)

    left_force = vectorized_left_thr_fun(thr_left, left_pwm_zero_point) * g # N
    right_force = vectorized_right_thr_fun(thr_right, right_pwm_zero_point) * g # N

    ## Calculate u and v
    
    # Integral method
    u_dot_integral = integrate(u_dot_corr, step)
    v_dot_integral = integrate(v_dot_corr, step)
     
    # Differentiate method
    dx = differentiate(xn, step)
    dy = differentiate(yn, step)

    vectorized_ned_to_bodyfixed_func = np.vectorize(ned_to_bodyfixed)
    dx_b, dy_b = vectorized_ned_to_bodyfixed_func(dx, dy, np.radians(yaw))
     
    # yaw_acc
    dr = differentiate(yaw, step)
 
    return pd.DataFrame({
        "origional_timestamp": t,
        "x_ned": xn,
        "y_ned": yn,
        "yaw": yaw,
        "surge": dx_b,
        "sway": dy_b,
        "yaw_rate": r,
        "surge_dot": u_dot_corr,
        "surge_dot_bias": np.full_like(u_dot_corr, u_bias),
        "sway_dot": v_dot_corr,
        "sway_dot_bias": np.full_like(v_dot_corr, v_bias),
        "yaw_acc": dr,
        "thr_left": thr_left,
        "thr_right": thr_right,
        "right_force": right_force,
        "left_force": left_force,
        "thr_left_pwm_zero_point": left_pwm_zero_point,
        "thr_right_pwm_zero_point": right_pwm_zero_point
    })

def fix_files(files: list):

    print(f"Extracting data, found {len(raw_data_files)} files")
    dataframes = []
    for file in files:
        print(f"{len(dataframes)}. doing file {file}")

        full_file_path = RAW_DATA_FOLDER.joinpath(file)
        raw_df = pd.read_csv(full_file_path)
        corrected_df = fix_all_data(raw_df)
        dataframes.append(corrected_df)
    
    full_df = pd.concat(dataframes)
    full_df.to_csv("full_dataset.csv")

 
def analyse(df: pd.DataFrame):
    t = df['timestamp(ms)'].to_numpy()
    u_dot = df['IMU.AccX'].to_numpy()
    v_dot = df['IMU.AccY'].to_numpy()


    duration = (t[-1] - t[0])/1000 # secounds
    # print("duration", duration)
    step = 0.1
    tt = np.arange(0, duration+step, step)
    # print(tt)


    u_bias = calculate_bias_acc(u_dot)
    u_dot_corr = correct_for_bias(u_dot, u_bias)

    v_bias = calculate_bias_acc(v_dot)
    v_dot_corr = correct_for_bias(v_dot, v_bias)

    plot_acc(t, u_dot, u_dot_corr, v_dot, v_dot_corr)

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


def analyse_all():
    for file in raw_data_files:
        print(file)
        full_file = RAW_DATA_FOLDER.joinpath(file)
        df = pd.read_csv(full_file)
        analyse(df)



if __name__ == '__main__':
    # analyse_all()

    fix_files(raw_data_files)




