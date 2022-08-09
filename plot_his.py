import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

C_DPI = 300

# C_DEFAULT_FONT_PATH = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
C_DEFAULT_FONT_PATH = 'C:\Windows\Fonts\Arial.ttf'

C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(fname=C_DEFAULT_FONT_PATH,
                                                  size=C_DEFAULT_FONT_SIZE)

Y_LABEL = r'n'
X_LABEL = r'length of saccades/ length of signal [%]'

LINEWIDTH = 1
LABELPAD = 2
TICKS_LEN = 2


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(7), cm2inch(7))


def axes_formatting(ax):
    """
    Details for the axis
    """
    # y_majors = np.arange(Y_AXIS["min"],
    #                      Y_AXIS["max"] + Y_AXIS["step"],
    #                      Y_AXIS["step"])
    #
    # x_majors = np.arange(X_AXIS["min"],
    #                      X_AXIS["max"] + X_AXIS["step"],
    #                      X_AXIS["step"])

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
    # ax.set_ylim(Y_AXIS["min"], Y_AXIS["max"])
    # ax.spines["left"].set_position(('data', 0))

    # labels = map(lambda x: "{:.0f}".format(x), y_majors)
    # ax.set_yticks(y_majors)
    # ax.set_yticklabels(labels)

    # Limits and ticks for x-axis
    # ax.set_xlim(X_AXIS["min"], X_AXIS["max"])
    # ax.spines["bottom"].set_position(('data', Y_AXIS["min"]))

    # labels = map(lambda x: "{:.0f}".format(x), x_majors)
    # ax.set_xticks(x_majors)
    # ax.set_xticklabels(labels)


    # Format labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)


def plot_hist(epoch_list, output_path):
    values = []
    fig, ax = plt.subplots(figsize=C_FIGSIZE)
    for e in epoch_list:
        values.append(len(e.saccades_idx)/e.signal.size * 100)

    ax.hist(values, color='k', fill=False)
    ax.set_xlabel(X_LABEL,
                  fontsize=C_DEFAULT_FONT_SIZE,
                  fontproperties=C_DEFAULT_FONT_PROP)

    ax.set_ylabel(Y_LABEL,
                  fontsize=C_DEFAULT_FONT_SIZE,
                  fontproperties=C_DEFAULT_FONT_PROP)

    prop = dict(left=0.13, top=0.99, bottom=0.15, right=0.99)
    plt.subplots_adjust(**prop)
    plt.savefig(output_path, dpi=C_DPI)
    plt.clf()
    plt.close()
