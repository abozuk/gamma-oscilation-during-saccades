import os
import shutil
import re
import json
import numpy as np
from wczytywanie_i_blinki_mne_numpy import wczytaj, detektor_bs
from epochs import EpochsListInCase
from plot_epochs import plot_ica_epochs
from plot_ch_epochs import plot_channels_epochs
from plot_his import plot_hist, plot_hist_inter
from time_freq import time_freq_spectogram, time_freq_scipy


def read_json(path):
    f = open(path)
    data = json.load(f)
    f.close()

    return data


def clean_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":

    # Tworzenie obrazków z epokami z ica[1]
    path = "Data"
    output = "output"
    clean_directory(output)
    file_list = os.listdir(path)
    epoch_service = EpochsListInCase()

    ica_order = read_json("ica_order.json")

    data_matrix = np.zeros((118, 19, 2, 60, 500))  # osoby, kanały, serie, częstość(zakres 0-60), czas
    names_to_idx = {}
    idx = 0
    files_names = []
    for f in file_list:
        pattern = "sub-.*_task_art_watch" + ".*\.vhdr$"

        if re.match(pattern, f):
            fname = os.path.join(path, f)
            print(fname)
            files_names.append(fname)
            try:
                signal_from_mne = wczytaj(fname)
            except:
                print("Nie udało się wczytać: ", fname)
                f = open(os.path.join(output, "raport.txt"), "w")
                f.write("Nie wczytano: {}\n".format(fname))
                f.close()
                continue

            names_to_idx[f'{fname[9:15]}'] = idx
            Fs = signal_from_mne.info['sfreq']
            ica = detektor_bs(signal_from_mne, "mne_lib")

            df = signal_from_mne.annotations.to_data_frame()
            try:
                ica_ch = ica_order[f]
            except:
                continue
            epoch_list = epoch_service.epochs_factory(df, signal_from_mne, ica, ica_ch)
            epoch_list.reverse()

            case = re.findall("-(.*?)t", f)[0]
            output_path = os.path.join(output, case[:-1])
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            plot_fname = "ica_"
            plot_fname += case
            plot_fname += re.findall("(watch\d*)?_run.*\.vhdr", f)[0]

            plot_fname += ".png"

            plot_path = os.path.join(output_path, plot_fname)

            # TUTAJ LISTA ODCINKÓW
            for s in [1, 2]:
                try:
                    list_of_sacceds_from_case = epoch_service.get_series(s)
                    print("Liczba odcinków:", len(list_of_sacceds_from_case), list_of_sacceds_from_case[0].shape)
                    P_all = time_freq_scipy(list_of_sacceds_from_case)
                    data_matrix[idx, :, s - 1, :, :] = P_all
                    plot_ica_epochs(epoch_list, Fs, plot_path, True)
                    plot_hist(epoch_list, os.path.join(output_path, plot_fname.replace("ica", "hist")))
                    plot_hist_inter(epoch_list,
                                    os.path.join(output_path,
                                                 plot_fname.replace("ica", "inter_saccades_hist")))
                except:
                    continue
            # plot_channels_epochs(epoch_list, Fs, plot_path)
            idx += 1
    np.save(os.path.join(output, "data_matrix.npy"), data_matrix)
    with open('names_idx.json', 'w') as f:
        json.dump(names_to_idx, f)