import scipy.signal as ss
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet, tfr_array_morlet

import matplotlib.font_manager as font_manager


C_DPI = 300

C_DEFAULT_FONT_PATH = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
#C_DEFAULT_FONT_PATH = 'C:\Windows\Fonts\Arial.ttf'

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

# def cm2inch(x):
#     return x * 0.39
#
#
# C_FIGSIZE = (cm2inch(7), cm2inch(28))

def time_freq_spectogram(signal, Fs, NFFT, plot_name):
    print("Spectogram")
    window = ss.hamming(NFFT)
    for ch in range(19):
        fig, ax = plt.subplots(figsize=C_FIGSIZE)
        P_all = 0
        for e_i in range(len(signal)):
            _s = signal[e_i][ch,:]
            P, freq, t, im1 = py.specgram(_s, NFFT=NFFT, Fs=Fs, window=window, noverlap=NFFT - 1,
                                          sides='onesided')

            P_all += np.abs(P)
        P_all = np.log(P_all/len(signal))
        P_all = P_all[2:8] #TODO
        print(P_all.shape, freq.shape, t.shape)


        # print(P_all.shape)

        py.imshow(P_all, aspect='auto', origin='lower',
              extent=(t[0], t[-1], 20, 80), interpolation='nearest')
        figname = plot_name.replace("ica", "ch_" + str(ch) + "_")
        py.show()
        # plt.imsave(figname, P_all, dpi=300)
        # plt.clf()


def cwt(x, MinF, MaxF, Fs, w=7.0, df=1.0, plot=False):
    '''w - parametr falki Morleta,
      wiąże się z jej częstościa centralną i skalą w następujacy sposób:
      f = 2*a*w / T
      gdzie: a-skala,  T-długość sygnału w sek.'''
    T = len(x) / Fs
    M = len(x)
    t = np.arange(0, T, 1. / Fs)
    freqs = np.arange(MinF, MaxF, df)
    P = np.zeros((len(freqs), M))
    X = np.fft.fft(x)  # transformacja sygnału do dziedziny czestosci
    for i, f in enumerate(freqs):  # petla po kolejnych czestosciach
        a = T * f / (2 * w)  # obliczenie skali dla danej czestosci
        psi = np.fft.fft(ss.morlet(M, w=w, s=a,
                                   complete=True))  # transformacja falki Morleta do dziedziny czestosci. W bardziej wydajnym kodzie moznaby zastosowac analityczna postac tej falki w dziedzinie czestosci.
        psi /= np.sqrt(np.sum(psi * psi.conj()))  # normalizacja energii falki
        CWT = np.fft.fftshift(np.fft.ifft(X * psi))
        P[i, :] = (CWT * CWT.conj()).real

    if plot:
        py.imshow(P, aspect='auto', origin='lower', extent=(0, T, MinF, MaxF))
        py.show()
    return P, freqs, t

def time_freq_scipy(signal):#, out_path, plot_name):
    widths = np.arange(5, 100)
    print("SIGNAL", len(signal))
    T = len(signal[0][0, :]) / 1000

    t = np.arange(0, T, 1. / 1000)
    freqs = np.arange(20, 80, 1.0)
    # py.figure()
    # py.imshow(np.abs(cwtmatr), extent=[0, 10, 0, 60], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
    #           vmin=-abs(cwtmatr).max())

    # PLOTTING TO NEW FUNCTION
    # fig, ax = plt.subplots(5,5,figsize=C_FIGSIZE)
    #
    # #preparing matrix of indexes, to make a kind of a topomap out of tf maps
    # idx_matrix = np.zeros((5, 5))
    # idx_matrix[1:4, :] = 1
    # idx_matrix[:, 1] = 1
    # idx_matrix[:, 3] = 1


    #nz = np.nonzero(idx_matrix)
    #########################################################################
    chosen_channels = ['Fp1', 'Fp2',
                       'F7', 'F3', 'Fz', 'F4', 'F8',
                       'T7', 'C3', 'Cz', 'C4', 'T8',
                       'P7', 'P3', 'Pz', 'P4', 'P8',
                       'O1', 'O2']
    powers = np.zeros((19,60,500))
    for ch in range(19):

        P_all = 0
        for e_i in range(len(signal)):
            #print(e_i)
            _s = signal[e_i][ch, :]
            # P, freq, t, im1 = py.specgram(_s, NFFT=NFFT, Fs=Fs, window=window, noverlap=NFFT - 1,
            #                               sides='onesided')
            cwtmatr, f, t = cwt(_s, 20, 80, 1000, 10)
            #print(cwtmatr.shape)
            #if ch == 0 and e_i == 0:
                #print( "czas: ", t, "\n","częstość: ", f)
            P_all += np.abs(cwtmatr)
        P_all = np.log(P_all / len(signal))
        #print("moc: ", P_all.shape)
        powers[ch,:,:] = P_all
        print("channel: ", ch+1)
        # P_all = P_all[2:8]  # TODO

        # print(P_all.shape)

        # py.imshow(P_all, aspect='auto', origin='lower',
        #           extent=(t[0], t[-1], 20, 80), interpolation='nearest')
        # figname = plot_name.replace("ica", "ch_" + str(ch) + "_")
        # py.imshow(P, aspect='auto', origin='lower', extent=(0, T, MinF, MaxF))
        # axes_formatting(ax)

        #PLOTTING TO NEW FUNCTION
        # ax[nz[0][ch], nz[1][ch]].imshow(P_all, extent=[0, 0.5, 20, 80],  aspect='auto', origin='lower')
        # ax[nz[0][ch], nz[1][ch]].title.set_text(chosen_channels[ch])




    # fig.suptitle(plot_name[4:-4])
    # #fig.tight_layout()
    # fig.subplots_adjust(top = 0.88, hspace = 0.5)
    #
    # #plt.show()
    # plt.savefig(out_path)
    #########################################################################
    #print(powers.shape)
    return powers
    # plt.clf()
    # plt.close()


# tu wersje z mne, ale mam problem z wczytaniem naszego sygnału jako "epoch":
# https://mne.tools/dev/auto_examples/time_frequency/time_frequency_simulated.html

def time_freq_mne(signal_epochs, freqs):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    all_n_cycles = [1, 3, freqs / 2.]
    for n_cycles, ax in zip(all_n_cycles, axs):
        power = tfr_morlet(signal_epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
        power.plot([0], baseline=(0., 0.1), mode='mean', axes=ax, show=False, colorbar=False)
        n_cycles = 'scaled by freqs' if not isinstance(n_cycles, int) else n_cycles
        ax.set_title('Sim: Using Morlet wavelet, n_cycles = %s' % n_cycles)
    plt.tight_layout()
