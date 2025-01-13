from matplotlib.text import FancyArrowPatch
import pandas as pd
import numpy as np
import fix_dataset
from pathlib import Path

from matplotlib import figure, pyplot as plt
from matplotlib.axes import Axes
# from scipy.signal import savgol_filter, butter, filtfilt
# from scipy.integrate import cumtrapz, simps

matlab_settings = {
    "title": dict(weight="bold", fontsize=12),
    "labels": dict(fontsize=12),
    "legend": dict(fontsize=10, fancybox=False, edgecolor="#585858")
}

def look_like_matlab(ax: Axes|None=None):
    ax_in_use = ax or plt.gca()
    plt.rcParams['font.family'] = "Helvetica"

    ax_in_use.spines["right"].set_visible(False)
    ax_in_use.spines["right"].set_color("#676767")
    ax_in_use.spines["top"].set_visible(False)
    ax_in_use.tick_params(direction="in", length=6, width=1)



if __name__ == "__main__":

    ax: Axes
    fig, ax = plt.subplots() # pyright: ignore
    look_like_matlab(ax)

    plt.show()


