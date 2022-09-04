import os
import shutil
import re
import json

from wczytywanie_i_blinki_mne_numpy import wczytaj, detektor_bs
from epochs import EpochsListInCase
from plot_epochs import plot_ica_epochs
from plot_his import plot_hist, plot_hist_inter
from time_freq import time_freq_spectogram, time_freq_scipy, time_freq_mne
import numpy as np

DATA_PATH = "Data"
OUTPUT_PATH = "output"
DATASET_TO_IGNORE = "to_ignore.txt"
ICA_ORDER = "ica_order.json"


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

def get_plot_name(fname):
    plot_fname = "ica_"
    plot_fname += case
    plot_fname += re.findall("(watch\d*)?_run.*\.vhdr", fname)[0]

    plot_fname += ".png"

    return plot_fname


if __name__ == "__main__":

    clean_directory(OUTPUT_PATH)
    f = open(os.path.join(OUTPUT_PATH, "raport.txt"), "w")
    f.close()

    file_list = os.listdir(DATA_PATH)
    epoch_service = EpochsListInCase()

    ica_order = read_json(ICA_ORDER)

    f = open(DATASET_TO_IGNORE)
    to_ignore_list = [l for l in f.read().split("\n")]

    for f in file_list:
        pattern = "sub-.*_task_art_watch" + ".*\.vhdr$"

        if re.match(pattern, f):
            fname = os.path.join(DATA_PATH, f)
            print(fname)
            try:
                case_name = re.findall("(?<=sub-).*(?=_run)", f)[0]
                if case_name in to_ignore_list:
                    print("Ignored:", case_name)
                    f = open(os.path.join(OUTPUT_PATH, "raport.txt"), "a")
                    f.write("Ignored: {}\n".format(fname))
                    f.close()
                    continue
                signal_from_mne = wczytaj(fname)
            except:
                print("Can't read: ", fname)
                f = open(os.path.join(OUTPUT_PATH, "raport.txt"), "a")
                f.write("Nie wczytano: {}\n".format(fname))
                f.close()
                continue

            Fs = signal_from_mne.info['sfreq']
            ica = detektor_bs(signal_from_mne, "mne_lib")

            df = signal_from_mne.annotations.to_data_frame()
            ica_ch = ica_order[f]
            epoch_list = epoch_service.epochs_factory(df, signal_from_mne, ica, ica_ch)
            epoch_list.reverse()

            case = re.findall("-(.*?)t", f)[0]
            print("case ", case[:-1])

            plot_fname = get_plot_name(f)
            plot_path = os.path.join(OUTPUT_PATH, plot_fname)

            # TUTAJ LISTA ODCINKÓW
            # freqs = np.arange(20., 80., 1.)
            for s in [1, 2]:
                # print("S: ", type(s))
                list_of_sacceds_from_case = epoch_service.get_series(s)
                print("Liczba odcinków:", len(list_of_sacceds_from_case), list_of_sacceds_from_case[0].shape)
                # time_freq_scipy(list_of_sacceds_from_case)
            plot_ica_epochs(epoch_list, Fs, plot_path, True)
            # plot_hist(epoch_list, os.path.join(output, plot_fname.replace("ica", "hist")))
            plot_hist_inter(epoch_list,
                            os.path.join(OUTPUT_PATH,
                                         plot_fname.replace("ica", "inter_saccades_hist")))
            # plot_channels_epochs(epoch_list, Fs, plot_path)
