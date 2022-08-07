import os
import re
import time

from wczytywanie_i_blinki_mne_numpy import wczytaj, detektor_bs
from epochs import epochs_factory, Epoch, ExtractEventInfo
from plot_epochs import plot_epochs

if __name__ == "__main__":

    # Tworzenie obrazków z epokami z ica[1]
    path = "./Data/"

    for f in os.listdir(path):
        pattern = "sub-.*_task_art_watch" + ".*\.vhdr$"

        if re.match(pattern, f):
            fname = os.path.join(path, f)
            signal_from_mne = wczytaj(fname)
            Fs = signal_from_mne.info['sfreq']
            ica = detektor_bs(signal_from_mne, "mne_lib")

            df = signal_from_mne.annotations.to_data_frame()
            epoch_list = epochs_factory(df, signal_from_mne, ica)
            print("AAAAAAA", len(epoch_list))
            epoch_list.reverse()

            plot_fname = "ica_"
            plot_fname += re.findall("-(.*?)t", f)[0]
            plot_fname += re.findall("_(run.*?)\.vhdr", f)[0]
            plot_fname+= ".png"
            print(plot_fname)

            output_path = os.path.join("./output/", plot_fname)
            plot_epochs(epoch_list, Fs, output_path)


