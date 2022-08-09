import copy
from find_saccades import get_min_diff, get_max_diff, find_saccades


class Epoch:
    def __init__(self, event_prop):
        # TODO better to have in dict ?
        self._idx = event_prop["event_idx"]
        self._onset = event_prop["onset"]
        self._duration = event_prop["duration"]
        self._series = event_prop["series"]
        self._img_no = event_prop["image_number"]
        self._st_idx = None
        self._end_idx = None
        self._sig = None
        self._Fs = None
        self._ica = None
        self._t = None
        self._saccades_idx = None

    def __str__(self):
        return "series: {}, image number: {}".format(self._series, self._img_no)

    def __repr__(self):
        return self._series + self._img_no

    @property
    def signal(self):
        return copy.copy(self._sig)

    @signal.setter
    def signal(self, sig_mne):
        self._Fs = sig_mne.info['sfreq']
        self._st_idx = sig_mne.time_as_index(self._onset)[0]
        self._end_idx = self._st_idx + int(self._duration * self._Fs)
        self._sig = sig_mne[:, self._st_idx:self._end_idx][0]
        self._t = sig_mne.times[self._st_idx:self._end_idx]

    @property
    def t(self):
        return copy.copy(self._t)

    @property
    def ica(self):
        return copy.copy(self._ica)

    @ica.setter
    def ica(self, ica):
        self._ica = ica[self._st_idx:self._end_idx]

    @property
    def saccades_idx(self):
        return copy.copy(self._saccades_idx)

    def find_saccades(self):
        sacc_sig = self._ica

        idx_to_diff = int(0.2 * self._Fs)
        sigFp = sacc_sig

        min_dff = get_min_diff(sigFp, idx_to_diff)
        max_dff = get_max_diff(sigFp, idx_to_diff)
        self._saccades_idx = find_saccades(sigFp, self._Fs, min_dff, max_dff)

        return sacc_sig, self._saccades_idx


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


def epochs_factory(df, sig_from_mne, ica):
    start_time = df.onset[0]
    end_time = df.onset.iloc[-1]
    epoch_list = []
    extractor = ExtractEventInfo(start_time, end_time)
    img_events = df.loc[df.description.str.contains("/P")]
    for idx, row in img_events.reindex().sort_index(ascending=False).iterrows():
        e = Epoch(extractor(idx, row))
        e.signal = sig_from_mne
        e.ica = ica[1][0][0]
        e.find_saccades()
        epoch_list.append(e)

    return epoch_list
