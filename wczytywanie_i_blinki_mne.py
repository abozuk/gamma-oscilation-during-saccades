#!/usr/bin/env python
#coding: utf-8

import mne
import os, re
import matplotlib.pyplot as plt
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
    
    eeg_mont = eeg.copy().set_montage(mapa)

    # ## REFERENCJA

    
    chosen_channels = ['Fp1', 'Fp2', 
                    'F7', 'F3', 'Fz', 'F4', 'F8', 
                    'T7', 'C3', 'Cz', 'C4', 'T8', 
                    'P7', 'P3', 'Pz', 'P4', 'P8', 
                    'O1', 'O2'] # (bez elektrod uszynych)



    #wybór 19 kanałóW
    eeg_20 = eeg_mont.copy()
    eeg_20.pick_channels(chosen_channels)

    #average reference as projection
    eeg_20.set_eeg_reference('average', projection=True)

    # ## FILTROWANIE
    #1 Hz bo tak robią w tutorialach mnne xd
    eeg_20.filter(1, None, l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', n_jobs=16)
    eeg_20.filter(None, 49, l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming', n_jobs=16)

    #eeg_20.plot(proj=True)
    
    return eeg_20


def detektor_bs(syg, option = 'numpy'): #/mne_library
    if option == 'numpy':
        print("Analiza z wykorzystaniem biblioteki numpy")
        
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

        ica.plot_components(inst=syg, psd_args={'fmax': 49.})
        ica.exclude = [0] #ręcznie wybrany komponent z mrugnięciami do usunięcia; przy większej ilości badanych można zastosowć bardziej automatyczny sposób
        reconst_syg = syg.copy()
        ica.apply(reconst_syg)

        syg.plot(title = "Sygnał przed" ) #sygnał przed rekonstrukcją

        
       
        reconst_syg.plot(title ="Sygnał po") #sygnał po usunięciu mrugnięć
        
        #del reconst_syg
        
       
# if ___name___ == '__main__':
for f in os.listdir('.'):
    pattern = "sub-ARZ000_task_art_watch"+".*\.vhdr$"
    if re.match(pattern, f):
        current_signal = wczytaj(f)
        #print("po pierwszej funkcji")
        detektor_bs(current_signal, 'mne_lib')
            
    


