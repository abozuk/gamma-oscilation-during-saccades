import os
import shutil
import re
import json

from wczytywanie_i_blinki_mne_numpy import wczytaj, detektor_bs
from epochs import EpochsListInCase
from plot_epochs import plot_ica_epochs
from plot_his import plot_hist, plot_hist_inter
from time_freq import time_freq_scipy
import numpy as np

DATA_PATH = "Data"
OUTPUT_PATH = "output"
DATASET_TO_IGNORE = "to_ignore.txt"
ICA_ORDER = "ica_order.json"
NUMBER_OF_DATASETS = 117


def read_json(path):
    json_file = open(path)
    data = json.load(json_file)
    json_file.close()

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

    # clean output dir
    clean_directory(OUTPUT_PATH)
    # create .txt file to storing information about the loaded data
    f = open(os.path.join(OUTPUT_PATH, "raport.txt"), "w")
    f.close()

    # list of files in the data folder
    file_list = os.listdir(DATA_PATH)

    # class instance to create custom epochs
    epoch_service = EpochsListInCase()

    # Manually selected ica channel for saccade detection
    ica_order = read_json(ICA_ORDER)

    # ignored cases/datasets
    f = open(DATASET_TO_IGNORE)
    to_ignore_list = [l for l in f.read().split("\n")]
    number_of_cases = NUMBER_OF_DATASETS - len(to_ignore_list)

    # cases, channels, series, frequencies (0-60 Hz), time
    data_matrix = np.zeros((number_of_cases, 19, 2, 60, 500))
    names_to_idx = {}
    idx = 0
    files_names = []

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

                # ---- reading signal by mne ----
                signal_from_mne = wczytaj(fname)
            except:
                print("Can't read: ", fname)
                f = open(os.path.join(OUTPUT_PATH, "raport.txt"), "a")
                f.write("Can't read: {}\n".format(fname))
                f.close()
                continue

            # sampling frequency
            Fs = signal_from_mne.info['sfreq']

            # 19 components ica
            ica = detektor_bs(signal_from_mne, "mne_lib")

            # information about the epoch in mne to data frame
            df = signal_from_mne.annotations.to_data_frame()
            ica_ch = ica_order[f]

            # list of epochs for the case
            epoch_list = epoch_service.epochs_factory(df, signal_from_mne,
                                                      ica, ica_ch)
            # reversing the order to be chronological in list
            epoch_list.reverse()

            # searching and creating names and paths
            case = re.findall("-(.*?)t", f)[0]
            print("case ", case[:-1])
            plot_fname = get_plot_name(f)
            plot_path = os.path.join(OUTPUT_PATH, plot_fname)

            # there are two series of images
            for s in [1, 2]:
                try:
                    # list of eras for a given series for a given case
                    list_of_sacceds_from_case = epoch_service.get_series(s)

                    P_all = time_freq_scipy(list_of_sacceds_from_case)
                    data_matrix[idx, :, s - 1, :, :] = P_all

                    # additional plots
                    plot_ica_epochs(epoch_list, Fs, plot_path, True)
                    plot_hist_inter(epoch_list,
                                    os.path.join(OUTPUT_PATH,
                                                 plot_fname.replace("ica", "inter_saccades_hist")))
                except:
                    continue
            idx += 1
