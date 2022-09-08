import numpy as np
import matplotlib.pyplot as plt
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


def plot_ica_epochs(epoch_list, Fs, output_path, saccades=False):
    n = len(epoch_list)
    n_rows = round(n / N_COLUMNS)
    n_plots = n_rows * N_COLUMNS
    t = np.arange(0, 25, 1 / Fs)

    fig, axs = plt.subplots(n_rows, N_COLUMNS, figsize=C_FIGSIZE)
    for i, e in enumerate(epoch_list):
        ax = axs[i // N_COLUMNS, i % N_COLUMNS]
        sig = e.ica
        ax.plot(t[:sig.shape[0]], sig, linewidth=LINEWIDTH, color='k')
        if saccades is True:
            starts, ends = e.loc_saccades_idx
            for s, e in zip(starts, ends):
                ax.plot(t[s:e], sig[s:e], linewidth=LINEWIDTH, color='red')

        axes_formatting(ax)

        if i >= n_plots - N_COLUMNS:
            ax.set_xlabel(X_LABEL,
                          fontsize=C_DEFAULT_FONT_SIZE,
                          fontproperties=C_DEFAULT_FONT_PROP)
        if i % N_COLUMNS == 0:
            ax.set_ylabel(Y_LABEL,
                          fontsize=C_DEFAULT_FONT_SIZE,
                          fontproperties=C_DEFAULT_FONT_PROP)

        ax.text(0.3, 5, e,
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=C_DEFAULT_FONT_SIZE,
                fontproperties=C_DEFAULT_FONT_PROP)

    if i < n_plots:
        for i in range(i + 1, n_plots):
            ax = axs[i // n_rows, i % N_COLUMNS]
            ax.axis('off')

    prop = dict(left=0.02, top=0.99, bottom=0.05, right=0.99, hspace=0.15, wspace=0.06)
    plt.subplots_adjust(**prop)

    plt.savefig(output_path, dpi=C_DPI)
    plt.cla()
    plt.close()
