import numpy as np
from pandas.core.internals.construction import treat_as_nested
# from pyproj import Proj, transform
import pandas as pd
from pathlib import Path
from navpy import lla2ned
from matplotlib import pyplot as plt

raw_data_folder = Path(__file__).resolve().parent.joinpath("raw_data")
sorted_data_folder = Path(__file__).resolve().parent.joinpath("sorted_data")

raw_data_files = [file.name for file in raw_data_folder.glob("*.csv")]

# Basestation-bench  (found on google maps)
REF_LON = 10.314897
REF_LAT = 63.401783
REF_ATT = 0

# df = pd.read_csv(raw_data_folder.joinpath(raw_data_files[0]))
# print(df.head())

def latlon_2_ned(lat, lon, att):
    ned = lla2ned(
        lat, lon, att,
        REF_LAT, REF_LON, REF_ATT
    )

    x_n = ned[:,0]
    y_n = ned[:,1]
    z_n = ned[:,2]

    return x_n, y_n, z_n

def derivate(array, dt):
    """Combines each i and i+1 elements, len out is therefore len(array-1)"""
    diff = np.diff(array)/dt
    out =  np.insert(diff, -1, diff[-1])
    return out

def integrate(array, dt):
    out = np.zeros(len(array))

    for i in range(len(out)-1):
        out[i+1]= (array[i+1]+array[i])*dt

    return out

def map_rcou_1_value(old_value):
    if old_value == 1505:
        return 0
    elif 1100 <= old_value < 1505:
        # Linear increase from 1100 to 1505 (0 to 5.63)
        return 5.63 * (1505 - old_value) / (1505 - 1100)
    elif 1505 < old_value <= 1900:
        # Linear decrease from 1505 to 1900 (0 to -2.81)
        return -2.81 * (old_value - 1505) / (1900 - 1505)
    elif old_value <= 1100:
        # For values under 1100 just map it to 5.63
        return 5.63
    elif old_value >= 1900:
        # For values over 1900 just map it to -2.81
        return -2.81

# RCOU_C3 = right motor
def map_rcou_3_value(old_value):
    if old_value == 1505:
        return 0
    elif 1100 <= old_value < 1505:
        # Linear decrease from 1505 to 1900 (0 to -2.81)
        return -2.81 * (1505 - old_value) / (1505 - 1100)
    elif 1505 < old_value <= 1900:
        # Linear increase from 1100 to 1505 (0 to 5.63)
        return 5.63 * (old_value - 1505 ) / (1900 - 1505)
    elif old_value <= 1100:
        # For values under 1100 just map it to 5.63
        return -2.81
    elif old_value >= 1900:
        # For values over 1900 just map it to -2.81
        return 5.63


def extract_data(df: pd.DataFrame):
    lon = df['GPS.Lng'] / 10000000
    lat = df['GPS.Lat'] / 10000000
    att = df['GPS.Alt']
    accx = df['IMU.AccX']
    accy = df['IMU.AccY']

    t = df['timestamp(ms)']/1000
    delta_t = np.diff(t)[0]

    x_n, y_n, z_n = latlon_2_ned(lat, lon, att)

    u = derivate(x_n, delta_t)
    v = derivate(y_n, delta_t)

    uu = integrate(accx, delta_t)
    vv = integrate(accy, delta_t)

    # Construct the dataframe
    df['Thrust.Left'] = df['RCOU.C1'].apply(map_rcou_1_value)
    df['Thrust.Right'] = df['RCOU.C3'].apply(map_rcou_3_value)

    df['Ned.x'] = x_n
    df['Ned.y'] = y_n
    df['Ned.z'] = z_n

    df['Integral.AccX.Surge'] = uu
    df['Integral.Accy.Sway'] = vv

    df['Differentiate.GPS.Surge'] = u
    df['Differentiate.GPS.Sway'] = v

    return df


for file in raw_data_files:
    raw_file_path = raw_data_folder.joinpath(file)
    raw_df = pd.read_csv(raw_file_path)
    print(raw_df.columns)
    extracted_df = extract_data(raw_df)
    print(extracted_df.head())
    extracted_df.to_csv(sorted_data_folder.joinpath(file))


# test = extract_data(df)
# print(test.columns)
# print(test.head())

