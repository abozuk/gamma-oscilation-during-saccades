import numpy as np

# from szukanie_sakkad import wczytaj
class Epoch:
    def __init__(self, event_prop):
        self._idx = event_prop["event_idx"]
        self._onset = event_prop["onset"]
        self._duration = event_prop["duration"]
        self._series = event_prop["series"]
        self._img_no = event_prop["image_number"]
        self._sig = None
        self._t = None

    def __str__(self):
        return self._series + self._img_no

    def get_epoch_signal(self, sig_mne):
        Fs = sig_mne.info['sfreq']
        st_idx = sig_mne.time_as_index(self._onset)[0]
        end_idx = st_idx + int(self._duration * Fs)
        print(st_idx, end_idx)
        self._sig = sig_mne[:, st_idx:end_idx][0]
        self._t = sig_mne.times[st_idx:end_idx]

    def plot(self):
        n = self._sig.shape[0]
        fig, axs = plt.subplots(4, 5)
        for i in range(n):
            axs[i // 5, i % 5].plot(self._t, self._sig[i, :])
        # prop
        # plt.savefig()
        plt.show()


class ExtractEventInfo:
    def __init__(self, start_time, end_time):
        self._start_t = start_time
        self._end_t = end_time
        self._previous_event_t = end_time

    def __call__(self, index, row):
        t = row.onset - self._start_t
        duration = self._previous_event_t - row.onset
        self._previous_event_t = row.onset
        img_descrp = row.description.partition("/P ")[-1]
        img_series, img_no = [img_descrp[0], img_descrp[1:]]
        props = self._to_dict(index,
                              t.total_seconds(), duration.total_seconds(),
                              img_series, img_no)
        return props

    def _to_dict(self, *args):
        _DICT_PROPS = ["event_idx", "onset", "duration", "series", "image_number"]
        props_dict = {}
        for key, value in zip(_DICT_PROPS, args):
            props_dict[key] = value
        return props_dict


def epochs_factory(df, signal_from_mne):
    start_time = df.onset[0]
    end_time = df.onset.iloc[-1]
    extractor = ExtractEventInfo(start_time, end_time)
    img_events = df.loc[df.description.str.contains("/P")]
    for idx, row in img_events.reindex().sort_index(ascending=False).iterrows():
        e = Epoch(extractor(idx, row))
        e.get_epoch_signal(signal_from_mne)
        e.plot()

path = 'sub-ARZ000_task_art_watch1_run-01.vhdr'
signal_mne = wczytaj(path)




