import copy
import numpy as np
from find_saccades import get_min_diff, get_max_diff, find_saccades


class Epoch:
    def __init__(self, event_prop):
        self._props = event_prop
        self._st_idx = None
        self._end_idx = None
        self._sig = None
        self._Fs = None
        self._ica = None
        self._t = None
        self._saccades_idx = None
        self._loc_saccades_idx = None
        self._inter_saccades_idx = None

    def __str__(self):
        return "series: {}, image number: {}".format(self.series,
                                                     self.image_number)

    def __repr__(self):
        return "series: {}, image number: {}".format(self.series,
                                                     self.image_number)

    @property
    def event_idx(self):
        return copy.copy(self._props["event_idx"])

    @property
    def onset(self):
        return copy.copy(self._props["onset"])

    @property
    def duration(self):
        return copy.copy(self._props["duration"])

    @property
    def series(self):
        return copy.copy(self._props["series"])

    @property
    def image_number(self):
        return copy.copy(self._props["image_number"])

    @property
    def signal(self):
        return copy.copy(self._sig)

    @signal.setter
    def signal(self, sig_mne):
        self._Fs = sig_mne.info['sfreq']
        self._st_idx = sig_mne.time_as_index(self.onset)[0]
        self._end_idx = self._st_idx + int(self.duration * self._Fs)
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

    @property
    def loc_saccades_idx(self):
        return copy.copy(self._loc_saccades_idx)

    @property
    def inter_saccades_idx(self):
        return copy.copy(self._inter_saccades_idx)

    def find_saccades(self):
        sacc_sig = self._ica
        self._saccades_idx = find_saccades(sacc_sig, self._Fs)

        self._locate_saccades()

        return sacc_sig, self._saccades_idx

    def _locate_saccades(self):
        ssacc = np.sort(self._saccades_idx)
        where_stops = np.where(np.diff(ssacc) > 1)[0]
        ends = ssacc[where_stops]
        np.append(ends, ssacc[-1])
        starts = [ssacc[0]]
        starts.extend(ssacc[where_stops + 1])
        starts = np.array(starts[:-1])
        self._loc_saccades_idx = (starts, ends)

        self._locate_inter_saccades(ssacc, where_stops, ends)
        return starts, ends

    def _locate_inter_saccades(self, ssacc, where_stops, ends):
        if self._loc_saccades_idx[0].size == 0:
            self._inter_saccades_idx = ([0], [self.signal.size])

            return 0
        elif self._loc_saccades_idx[0][0] == 0:
            inter_starts = [0]
        else:

            inter_starts = []

        inter_starts.extend(ends)
        inter_ends = ssacc[where_stops + 1]
        np.append(inter_ends, self._loc_saccades_idx[0][-1])
        self._inter_saccades_idx = (inter_starts, inter_ends)


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

    @staticmethod
    def _to_dict(*args):
        _DICT_PROPS = ["event_idx", "onset", "duration", "series", "image_number"]
        props_dict = {}
        for key, value in zip(_DICT_PROPS, args):
            props_dict[key] = value
        return props_dict


class EpochsListInCase:
    """
    The purpose of this class is to
    load and maintain the epochs in single case.

    Only one instance of class in pipeline needed.

    methods:
        epochs_factory()

        get series()

    """
    def __init__(self):
        self._epoch_list = []
        self._n_epochs = 0

    def __getitem__(self, i):
        return self._epoch_list[i]

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i < self._n_epochs:
            self.__i += 1
            return self._epoch_list[self.__i - 1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self._epoch_list)

    def epochs_factory(self, df, sig_from_mne, ica, ica_ch):
        start_time = df.onset[0]
        end_time = df.onset.iloc[-1]

        # clear epoch list
        self._epoch_list = []
        extractor = ExtractEventInfo(start_time, end_time)
        img_events = df.loc[df.description.str.contains("/P")]
        for idx, row in img_events.reindex().sort_index(ascending=False).iterrows():
            e = Epoch(extractor(idx, row))
            e.signal = sig_from_mne
            e.ica = ica[ica_ch][0][0]

            e.find_saccades()
            self._epoch_list.append(e)

        self._n_epochs = len(self)
        return self._epoch_list

    def get_series(self, n_series, section_len=500):
        list_of_array = []
        for e in self._epoch_list:
            if int(e.series) == n_series:
                starts, ends = e.inter_saccades_idx
                for st_idx, end_idx in zip(starts, ends):
                    _len = np.abs(st_idx - end_idx)
                    if _len > section_len:
                        _arr = np.zeros((e.signal.shape[0], section_len))
                        for ch in range(_arr.shape[0]):
                            _s = e.signal[ch, st_idx:st_idx + section_len]
                            _arr[ch, :] = _s

                        list_of_array.append(_arr)

        return list_of_array



                # zwróć w macierzy odpowiednie odcinki czy jakoś tak


