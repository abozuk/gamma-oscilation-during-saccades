#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:25:17 2022

@author: justyna
"""

#!/usr/bin/env python
#coding: utf-8

import mne
import os, re
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import numpy as np 
from mne.preprocessing import ICA



def wczytaj(filename):#/mne_lib
    #po kolei dla plików z folderu

    #wczyttywanie i wstępna obróbka
    #print(filename)
    eeg = mne.io.read_raw_brainvision(filename,preload = True)
    # plt.figure()
    # eeg.plot()
    # plt.show()
    #eeg info
    #print(eeg.info)
    
    # ## MONTAŻ


    mapa = mne.channels.make_standard_montage('standard_1020')
    #plotowanie montażu
    #fig = mapa.plot(kind='3d')
    #fig.gca().view_init(azim=70, elev=15)  # set view angle
    #mapa.plot(kind='topomap', show_names=True)

    #print(mapa)

    #--------------interpolacja na razie tylko do sprawdzenia złego sygnału
    # eeg.info['bads'].append('F8')
    # eeg.info['bads'].append('T7')
    print(eeg.info['bads'])
    
    eeg_mont = eeg.copy().set_montage(mapa)

    # ## REFERENCJA

    
    chosen_channels = ['Fp1', 'Fp2', 
                    'F7', 'F3', 'Fz', 'F4', 'F8', 
                    'T7', 'C3', 'Cz', 'C4', 'T8', 
                    'P7', 'P3', 'Pz', 'P4', 'P8', 
                    'O1', 'O2'] # (bez elektrod uszynych)



    #wybór 19 kanałóW
    eeg_20 = eeg_mont.copy()
    eeg_20.pick_channels(chosen_channels) #TODO!!!

    #--------------interpolacja cz.2. na razie tylko dla przykładowego sygnału
    # eeg_20.interpolate_bads(reset_bads=False)
    print("przed referencją")


    #average reference as projection
    eeg_20.set_eeg_reference('average', projection=True)

    # ## FILTROWANIE
    #1 Hz bo tak robią w tutorialach mnne xd
    eeg_20.filter(1, None, l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', n_jobs=16)
    eeg_20.notch_filter(freqs=50, filter_length='auto', phase='zero', method="iir")

    eeg_20.filter(None, 80, l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', n_jobs=16)

    #eeg_20.plot(proj=True)
    
    return eeg_20


def detektor_bs(syg, option = 'numpy'): #/mne_library
    if option == 'numpy':
        print("Analiza z wykorzystaniem biblioteki numpy")
        
        #biore Fp1 i Fp2 bo są najbliżej oczu
        syg1 = syg.copy().pick_channels(['Fp1'])
        syg1_m = syg1.get_data().flatten() #robi macierz
        print(syg1_m.shape)
        
        odch_stand1 = np.std(syg1_m)
        peaks1, peaks_dict1 = find_peaks(syg1_m*(-1), prominence = odch_stand1)
        #print(peaks1)
        print(len(peaks1))
        
        syg2 = syg.copy().pick_channels(['Fp2'])
        syg2_m = syg2.get_data().flatten() #robi macierz
        
        odch_stand2 = np.std(syg2_m)
        peaks2, peaks_dict2 = find_peaks(syg2_m*(-1), prominence = odch_stand2)
        print(len(peaks2))
        #print(peaks2)
        
        blinki = []
        for i in peaks1:
            for j in peaks2:
                if i == j:
                    blinki.append(i)
                    
        print(blinki)
        print(len(blinki)) #jest ich 70, czyli się zgadza, bo mrugamy około 17 razy na minutę
        
        pozorne_blinki = np.concatenate((peaks1, peaks2))
        print(len(pozorne_blinki))
        
        
        Fs = syg.info['sfreq'] #czestosc probkowania
        t = syg.times # w sek
        czas = len(t)/Fs
        print(czas, 's')
        
        blinki = np.array(blinki)
        
        chosen_channels = ['Fp1', 'Fp2', 
                    'F7', 'F3', 'Fz', 'F4', 'F8', 
                    'T7', 'C3', 'Cz', 'C4', 'T8', 
                    'P7', 'P3', 'Pz', 'P4', 'P8', 
                    'O1', 'O2']
        
        syg_caly = syg.copy().pick_channels(chosen_channels)
        syg_caly_m = syg_caly.get_data().flatten()
        syg_caly_m = np.array(syg_caly_m)
        syg_caly_m = syg_caly_m.reshape((19,298720))
        
        print(syg_caly_m)
        print(syg_caly_m.shape)
        
    # def plot_channels(signal, t, channel_names):
    #         plt.figure(figsize = (9,9))
    #         n_plots = signal.shape[0]
    #         for i in range(signal.shape[0]):
    #             plt.subplot(n_plots, 1, i+1)
    #             plt.plot(t, signal[i,:])
    #             plt.title(channel_names[i])
    #
    #         plt.subplots_adjust(hspace=.9)
    #         plt.show()
    #
    #     plot_channels(syg_caly_m, t, chosen_channels) #przed usunięciem blinków
    #
    #     for k in range(19):
    #         for idx in blinki:
    #             syg_caly_m[k, idx] = 0
    #
    #     print(syg_caly_m)
    #     print(syg_caly_m.shape)
    #
    #
    #     plot_channels(syg_caly_m, t, chosen_channels) #po usunięciu blinków
        
        
        #print(sigFp)
        #plt.plot(t, sigFp)
        #plt.show()
        
        
        
    elif option == 'mne_lib':
        print("Analiza z wykorzystaniem biblioteki mne")
        
        #usuwanie mrugnięć z sygnału

        raw = syg

        
        picks_eeg = mne.pick_types(syg.info, meg=False, eeg=True, eog=False,
                                   stim=False, exclude='bads')
        method = 'fastica' #wybrana metoda ICA
        
        
        n_components = 19
        decim = 3  
        random_state = 15
        
        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        
        reject = dict(mag=5e-10, grad=4000e-11)
        ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)

        # ica.plot_components(inst=syg, psd_args={'fmax': 49.})
        ica.exclude = [0] #ręcznie wybrany komponent z mrugnięciami do usunięcia; przy większej ilości badanych można zastosowć bardziej automatyczny sposób
        reconst_syg = syg.copy()
        ica.apply(reconst_syg)

        # syg.plot(title = "Sygnał przed" ) #sygnał przed rekonstrukcją

        
       
        # reconst_syg.plot(title ="Sygnał po") #sygnał po usunięciu mrugnięć
        return ica.get_sources(syg)
        
        #del reconst_syg
        
       
if __name__ == '__main__':
    for f in os.listdir('.'):
        #ARZ000
        pattern = "sub-CKA387_task_art_watch"+".*\.vhdr$"
        if re.match(pattern, f):
            current_signal = wczytaj(f)
            print("po pierwszej funkcji")
            a = detektor_bs(current_signal, 'mne_lib')
            #detektor_bs(current_signal, 'numpy')
            print(a)
            