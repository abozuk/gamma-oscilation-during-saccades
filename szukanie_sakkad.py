#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:25:17 2022

@author: justyna
"""

#!/usr/bin/env python
#coding: utf-8

import mne, io
import os, re
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import numpy as np 
from mne.preprocessing import ICA
import scipy.stats as st
import pylab as py
import random


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
        
        def plot_channels(signal, t, channel_names):
            plt.figure(figsize = (9,9))
            n_plots = signal.shape[0]
            for i in range(signal.shape[0]):
                plt.subplot(n_plots, 1, i+1)
                plt.plot(t, signal[i,:])
                plt.title(channel_names[i])
        
            plt.subplots_adjust(hspace=.9)
            plt.show()
            
        plot_channels(syg_caly_m, t, chosen_channels) #przed usunięciem blinków
        
        for k in range(19):
            for idx in blinki:
                syg_caly_m[k, idx] = 0
            
        print(syg_caly_m)
        print(syg_caly_m.shape)
        
        
        plot_channels(syg_caly_m, t, chosen_channels) #po usunięciu blinków
        
        
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
        
        ica.exclude = [1] #ręcznie wybrany komponent z mrugnięciami do usunięcia; przy większej ilości badanych można zastosowć bardziej automatyczny sposób
        reconst_syg2 = reconst_syg.copy()
        ica.apply(reconst_syg2)
      
       
        # reconst_syg2.plot(title ="Sygnał po2") #sygnał po usunięciu mrugnięć
        
        
        # mne.viz.plot_ica_overlay(ica, reconst_syg, exclude=[1], picks=None, start=None, stop=None, title=None, show=True, n_pca_components=None)
        
        
        picks1 = mne.pick_types(reconst_syg.info, eeg = True)
        data1, times = reconst_syg[picks1,:]  
        
        picks2 = mne.pick_types(reconst_syg2.info, eeg = True)
        data2, times = reconst_syg2[picks2,:]  
        #del reconst_syg
        return ica, data1, data2, times
        
       
# if ___name___ == '__main__':
for f in os.listdir('.'):
    pattern = "sub-ARZ000_task_art_watch"+".*\.vhdr$"
    if re.match(pattern, f):
        current_signal = wczytaj(f)
        #print("po pierwszej funkcji")
        #detektor_bs(current_signal, 'mne_lib')
        i, d1,d2,t = detektor_bs(current_signal, 'mne_lib')
        y_diff = np.abs(d1[1,:]-d2[1,:]) #moduł różnicy miedzy sygnałem z sakkadami i sygnałem, w którym je usunęlismy
        
        #szukanie wartosci powyzej których uznajemy cos za sakkady
        alfa = 0.3 # trzeba ustalic tę wartosc
        lo = st.scoreatpercentile(y_diff, per = alfa/2*100)# nie wiem czy wystarczy na podstawie tego czy trzeba jakiegos bootstrapa
        hi = st.scoreatpercentile(y_diff, per = (1-alfa/2)*100)
        print('przedzial ufnosci: %(lo).8f - %(hi).8f\n'%{'lo':lo,'hi':hi})
        szer_binu = (hi-lo)/10
        biny = np.arange(lo-10*szer_binu, hi+11*szer_binu, szer_binu)
        py.figure()
        (n,y,patch) = py.hist(y_diff,bins = biny )
        py.plot([lo, lo] , [0, np.max(n)] ,'r' )
        py.plot([hi,hi],[0, np.max(n)],'r')
        py.show()
        
        
        #detekcja sakkad
        saccades = d1[1,:].copy()
        saccades[y_diff<hi] = np.nan # bieżemy tylko te które są powyżej wartosci odciecia hi
        
        plt.figure()
        #plt.hist(y_diff, bins = 100) #histogram żeby mniej więcej okreslic które wartosci odcinamy (może dodać automatyczne odcinanie jakiegos procentu?)
        
        # do rysowania biorę pierwsze 60s
        plt.plot(t[:60*1000],d1[1,:60*1000], "red") # sygnał Fp1 z usuniętymi blinkami ale z sakkadami
        plt.plot(t[:60*1000],d2[1,:60*1000], "black") # sygnał Fp1 z usuniętymi blinkami i sakkadami
        plt.plot(t[:60*1000],saccades[:60*1000], "blue") # próba zaznaczenia sakkad
        plt.legend(["syg z sakkadami", "syg bez sakkad", "same sakkady"])
        plt.show()

            
