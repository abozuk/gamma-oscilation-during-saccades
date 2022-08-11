import scipy.signal as ss
import pylab as py
import numpy as np
import matplotlib as plt
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet, tfr_array_morlet


def time_freq_spectogram(signal, Fs, NFFT):
    window = ss.hamming(NFFT)

    py.figure()
    P, freq, t, im1 = py.specgram(signal, NFFT=len(h), Fs=Fs, window=window, noverlap=NFFT - 1, sides='onesided')
    py.imshow(P, aspect='auto', origin='lower',
              extent=(t[0] - (NFFT / 2) / Fs, t[-1] - (NFFT / 2) / Fs, freq[0], freq[-1]), interpolation='nearest')


def time_freq_scipy(signal):
    widths = np.arange(5, 100)
    cwtmatr = ss.cwt(signal, ss.morlet2, widths)
    py.figure()
    py.imshow(cwtmatr.real, extent=[0, 10, 0, 60], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
              vmin=-abs(cwtmatr).max())


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
