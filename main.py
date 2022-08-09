import os
import shutil
import re

from wczytywanie_i_blinki_mne_numpy import wczytaj, detektor_bs
from epochs import epochs_factory
from plot_epochs import plot_ica_epochs
from plot_ch_epochs import plot_channels_epochs
from plot_his import plot_hist


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
    path = "./Data"
    output = "./output"
    clean_directory(output)
    file_list = os.listdir(path)

    for f in file_list:
        pattern = "sub-.*_task_art_watch" + ".*\.vhdr$"

        if re.match(pattern, f):
            fname = os.path.join(path, f)
            try:
                signal_from_mne = wczytaj(fname)
            except:
                print("Nie udało się wczytać: ", fname)
                f = open(os.path.join(output, "raport.txt"), "w")
                f.write("Nie wczytano: {}\n".format(fname))
                f.close()
                continue

            Fs = signal_from_mne.info['sfreq']
            ica = detektor_bs(signal_from_mne, "mne_lib")

            df = signal_from_mne.annotations.to_data_frame()
            epoch_list = epochs_factory(df, signal_from_mne, ica)
            epoch_list.reverse()

            case = re.findall("-(.*?)t", f)[0]
            output_path = os.path.join(output, case[:-1])
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            plot_fname = "ica_"
            plot_fname += case
            plot_fname += re.findall("(watch\d*_run.*)?\.vhdr", f)[0]

            plot_fname += ".png"

            plot_path = os.path.join(output_path, plot_fname)
            plot_ica_epochs(epoch_list, Fs, plot_path, True)
            plot_hist(epoch_list, os.path.join(output_path, plot_fname.replace("ica", "hist")))
            plot_channels_epochs(epoch_list, Fs, plot_path)
