import scipy.signal as ss
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet, tfr_array_morlet

from main import read_json

import matplotlib.font_manager as font_manager

C_DPI = 300

# C_DEFAULT_FONT_PATH = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
C_DEFAULT_FONT_PATH = 'C:\Windows\Fonts\Arial.ttf'

C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(fname=C_DEFAULT_FONT_PATH,
                                                  size=C_DEFAULT_FONT_SIZE)

Y_LABEL = r'ica'
X_LABEL = r'time [s]'

LINEWIDTH = 1
LABELPAD = 2
TICKS_LEN = 2

X_AXIS = {"min": 0, "max": 25, "step": 5}  # values for ticks
Y_AXIS = {"min": -5, "max": 5, "step": 5}

N_COLUMNS = 3


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(50), cm2inch(19))


def axes_formatting(ax):
    """
    Details for the axis
    """
    y_majors = np.arange(Y_AXIS["min"],
                         Y_AXIS["max"] + Y_AXIS["step"],
                         Y_AXIS["step"])

    x_majors = np.arange(X_AXIS["min"],
                         X_AXIS["max"] + X_AXIS["step"],
                         X_AXIS["step"])

    # Distance between ticks and label of ticks
    ax.tick_params(
        axis="y",
        which="major",
        pad=LABELPAD,
        length=TICKS_LEN,
        width=1,
        left="off",
        labelleft="off",
        direction="in")

    ax.tick_params(
        axis='x',
        which='major',
        pad=LABELPAD,
        length=TICKS_LEN,
        width=1,
        left="off",
        labelleft="off",
        direction="in")

    # Make rightline invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Limits and ticks for y-axis
    ax.set_ylim(Y_AXIS["min"], Y_AXIS["max"])
    ax.spines["left"].set_position(('data', 0))

    labels = map(lambda x: "{:.0f}".format(x), y_majors)
    ax.set_yticks(y_majors)
    ax.set_yticklabels(labels)

    # Limits and ticks for x-axis
    ax.set_xlim(X_AXIS["min"], X_AXIS["max"])
    ax.spines["bottom"].set_position(('data', Y_AXIS["min"]))

    labels = map(lambda x: "{:.0f}".format(x), x_majors)
    ax.set_xticks(x_majors)
    ax.set_xticklabels(labels)

    # Format labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)


def power_density_map(data_path, idx_names_path):
    p_all = np.load(data_path)
    case_names = read_json(idx_names_path)

    idx_matrix = np.zeros((5, 5))
    idx_matrix[1:4, :] = 1
    idx_matrix[:, 1] = 1
    idx_matrix[:, 3] = 1

    nz = np.nonzero(idx_matrix)

    chosen_channels = ['Fp1', 'Fp2',
                       'F7', 'F3', 'Fz', 'F4', 'F8',
                       'T7', 'C3', 'Cz', 'C4', 'T8',
                       'P7', 'P3', 'Pz', 'P4', 'P8',
                       'O1', 'O2']

    for i, case in enumerate(case_names):
        for s in [0, 1]:
            out_path = "output/" + case + "_seria_" + str(s) + ".png"
            fig, axs = plt.subplots(5, 5, figsize=C_FIGSIZE)
            for ch in range(p_all.shape[1]):
                # PLOTTING TO NEW FUNCTION
                axs[nz[0][ch], nz[1][ch]].imshow(p_all[i,ch,s,:], extent=[0, 0.5, 20, 80],
                                                 aspect='auto', origin='lower')
                axs[nz[0][ch], nz[1][ch]].title.set_text(chosen_channels[ch])


            fig.subplots_adjust(top = 0.88, hspace = 0.5)
            plt.savefig(out_path)
            plt.cla()
            plt.close()

    # mean
    p_mean = np.mean(p_all, axis=0)
    for s in [0, 1]:
        out_path = "output/" + "mean_seria_" + str(s) + ".png"
        fig, axs = plt.subplots(5, 5, figsize=C_FIGSIZE)
        for ch in range(p_all.shape[1]):
            # PLOTTING TO NEW FUNCTION
            axs[nz[0][ch], nz[1][ch]].imshow(p_mean[ch, s, :], extent=[0, 0.5, 20, 80],
                                             aspect='auto', origin='lower')
            axs[nz[0][ch], nz[1][ch]].title.set_text(chosen_channels[ch])

        fig.subplots_adjust(top=0.88, hspace=0.5)
        plt.savefig(out_path)
        plt.cla()
        plt.close()

power_density_map("output/data_matrix.npy", "output/names_idx.json")
