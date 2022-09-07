import numpy as np
import matplotlib.pyplot as plt
import os
from stats import stats
from average_tf_map import average_tf

from main import read_json

import matplotlib.font_manager as font_manager

C_DPI = 300

# C_DEFAULT_FONT_PATH = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
C_DEFAULT_FONT_PATH = 'C:\Windows\Fonts\Arial.ttf'

C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(fname=C_DEFAULT_FONT_PATH,
                                                  size=C_DEFAULT_FONT_SIZE)

Y_LABEL = r'Częstość [Hz]'
X_LABEL = r'Czas [s]'

LINEWIDTH = 1
LABELPAD = 2
TICKS_LEN = 2

X_AXIS = {"min": 0, "max": 0.5, "step": 0.1}  # values for ticks
Y_AXIS = {"min": 20, "max": 80, "step": 20}

N_COLUMNS = 3


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(19), cm2inch(13))


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

    # # Make rightline invisible
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    #
    # Limits and ticks for y-axis
    # ax.set_ylim(Y_AXIS["min"], Y_AXIS["max"])
    ax.spines["left"].set_position(('data', 0))

    labels = map(lambda x: "{:.0f}".format(x), y_majors)
    ax.set_yticks(y_majors)
    ax.set_yticklabels(labels)

    # Limits and ticks for x-axis
    # ax.set_xlim(X_AXIS["min"], X_AXIS["max"])
    ax.spines["bottom"].set_position(('data', Y_AXIS["min"]))

    labels = map(lambda x: "{:.1f}".format(x) if x != 0 else "{:.0f}".format(x), x_majors)
    ax.set_xticks(x_majors)
    ax.set_xticklabels(labels)

    # Format labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)


def power_density_map(p, ch_number, nz, chosen_channels, out_path):
    fig, axs = plt.subplots(5, 5, figsize=C_FIGSIZE)
    for ch in range(ch_number):
        ax = axs[nz[0][ch], nz[1][ch]]

        im = ax.imshow(p[ch], extent=[0, 0.5, 20, 80],
                       aspect='auto', origin='lower')  # extent=[0, 0.5, 0, 60]
        # plt.colorbar(im, cax=ax)

        ax.title.set_text(chosen_channels[ch])
        ax.title.set_fontproperties(C_DEFAULT_FONT_PROP)
        axes_formatting(ax)

    for ax_y in [0, 2, 4]:
        axs[3, ax_y].set_xlabel(X_LABEL,
                                fontsize=C_DEFAULT_FONT_SIZE,
                                fontproperties=C_DEFAULT_FONT_PROP)
        for ax_x in [0, 4]:
            axs[ax_x, ax_y].axis('off')

    for ax_y in [1,3]:
        axs[4, ax_y].set_xlabel(X_LABEL,
                                fontsize=C_DEFAULT_FONT_SIZE,
                                fontproperties=C_DEFAULT_FONT_PROP)
    for ax_x in range(1, 4):
        axs[ax_x, 0].set_ylabel(Y_LABEL,
                                fontsize=C_DEFAULT_FONT_SIZE,
                                fontproperties=C_DEFAULT_FONT_PROP)

    for ax_x in [0, 4]:
        axs[ax_x, 1].set_ylabel(Y_LABEL,
                                fontsize=C_DEFAULT_FONT_SIZE,
                                fontproperties=C_DEFAULT_FONT_PROP)



    prop = dict(left=0.055, top=0.96, bottom=0.075, right=0.99, hspace=0.55, wspace=0.3)
    plt.subplots_adjust(**prop)
    plt.savefig(out_path, dpi=C_DPI)
    plt.cla()
    plt.close()


def get_power_plots(avg_tf, rejected_fdr):
    # p_all = np.load(data_path)
    # case_names = read_json(idx_names_path)

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

    tf_diff = avg_tf[:,:,0,:,:] - avg_tf[:,:,1,:,:]
    print(tf_diff.shape)
    tf_diff_avg = np.mean(tf_diff, axis=0)
    print(tf_diff_avg.shape)
    out_path= "output/00.png"
    power_density_map(tf_diff_avg, tf_diff_avg.shape[0], nz, chosen_channels, out_path)

    # for i, case in enumerate(case_names):
    #     for s in [0, 1]:
    #         out_path = "output/" + case + "_seria_" + str(s) + ".png"
    #         p = p_all[i, :, s, :]
    #         power_density_map(p, p_all.shape[1], nz, chosen_channels, out_path)

    # mean
    # p_mean = np.mean(p_all, axis=0)
    # print(p_mean.shape)
    # for s in [0, 1]:
        # out_path = "output/" + "mean_seria_" + str(s) + ".png"
        # power_density_map(p_mean[:,s], p_all.shape[1], nz, chosen_channels, out_path)

    # dodać obrysy, przezroczystości
    # ujednolicić pomiędzy seriami, wartości różnicy pod maską nie gęstość mocy
    # normalizowane do odchyleń różnic ewentualnie


    # policzyć różnice pomiedzy warunkami wewnątrz osobniczo i uśrednić

    # mapka różnic
    #rozmiar maciery - różnice
    # beza absa!!!

if __name__ == "__main__":
    filename = os.path.join("output", "data_matrix.npy")
    average = average_tf(filename, 5, 25)  # 19 kan, 2 serie, 60 x 500
    print(average.shape)
    # uśrednianie po osobach
    # avg1 = np.mean(average[:, :, 0, :, :], axis=0)
    # avg2 = np.mean(average[:, :, 1, :, :], axis=0)

    # H0: średnie w obu seriach są takie same
    p_val, fdr_rejected = stats(average)
    get_power_plots(average, fdr_rejected)
