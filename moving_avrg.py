import numpy as np


def moving_average(signal, len_single_avrg, overlap=1):
    """
    signal - zapis sygnału
    len_single_avrg - długość w liczbie próbek z jakiego okresu liczona jest średnia
    overlap - liczba nakładający się próbek sygnału
    Funkcja zwraca wektor średnich dla len_single_avrg liczby próbek wstecz
    """
    sig_len = len(signal)
    samples_to_avrg = int(len_single_avrg - overlap)

    mov_avrgs = np.zeros(sig_len)
    mov_avrgs[:len_single_avrg] = np.mean(signal[:len_single_avrg])

    for i in range(samples_to_avrg, sig_len - len_single_avrg, samples_to_avrg):
        mov_avrgs[i: i + len_single_avrg] = np.mean(signal[i: i + len_single_avrg])

    for i in range(len_single_avrg, 0, -1):
        mov_avrgs[-i:i] = np.mean(signal[-i:])

    return mov_avrgs
