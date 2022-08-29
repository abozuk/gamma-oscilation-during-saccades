import numpy as np


def get_min_diff(sig, idx_to_diff):
    diff_sig = np.abs(sig[:-idx_to_diff] - sig[idx_to_diff:])
    return np.quantile(diff_sig, 0.67)


def get_max_diff(sig, idx_to_diff):
    diff_sig = np.abs(sig[:-idx_to_diff] - sig[idx_to_diff:])
    return np.quantile(diff_sig, 0.97)


def moving_average(signal, len_single_avrg, overlap=1, extend=True):
    '''
    signal - zapis sygnału
    len_single_avrg - długość w liczbie próbek z jakiego okresu liczona jest średnia
    overlap - liczba nakładający się próbek sygnału
    Funkcja zwraca wektor średnich dla len_single_avrg liczby próbek wstecz

    '''
    sig_len = len(signal)
    samples_to_avrg = int(len_single_avrg - overlap)

    mov_avrgs = np.zeros(sig_len)
    mov_avrgs[:len_single_avrg] = np.mean(signal[:len_single_avrg])

    for i in range(samples_to_avrg, sig_len - len_single_avrg, samples_to_avrg):
        mov_avrgs[i: i + len_single_avrg] = np.mean(signal[i: i + len_single_avrg])

    for i in range(len_single_avrg, 0, -1):
        mov_avrgs[-i:i] = np.mean(signal[-i:])

    return mov_avrgs


def find_saccades(sig, Fs, min_diff=25, max_diff=80):
    """

    :param sig:
    :param Fs:
    :param min_diff:
    :param max_diff:
    :return:
    """

    step = int(0.02 * Fs)  # bo minimalna długość sakkady

    avg_len = int(0.4 * Fs)
    mov_avg = moving_average(sig, avg_len)
    mov_avg_diff = sig - mov_avg
    beg_idx = np.where(np.abs(mov_avg_diff) > min_diff - 0.02)[0]

    new_ma = moving_average(sig, step, 1)
    new_ma_diff = np.diff(new_ma)

    diff = np.diff(sig)

    final_saccades = []
    temp_points = []
    for i in range(sig.shape[0] // avg_len + 1):
        sacc = np.intersect1d(beg_idx[beg_idx > i * avg_len],
                              beg_idx[beg_idx < (i + 1) * avg_len])
        ma_diff = mov_avg[sacc] - sig[sacc]

        if np.any(ma_diff < 0) and np.any(ma_diff > 0):

            new_point = int(np.median(sacc))

            # w otoczeniu 0.2 s znajdź, gdzie pochodna jest nawiększa i wybierz ten punkt
            t_range = int(0.2 * Fs)
            if new_point - t_range < 0:
                new_point = np.argmax(np.abs(diff[:new_point + t_range]))

            elif new_point >= len(diff) - t_range:
                new_point = np.argmax(np.abs(diff[new_point - t_range:])) + new_point - t_range

            else:
                new_point = np.argmax(np.abs(diff[new_point - t_range:new_point + t_range])) + new_point - t_range

            temp_points.append(new_point)

        if np.abs(mov_avg[i * avg_len] - mov_avg[i * avg_len - step]) > max_diff:
            new_point = int(i * avg_len - step)
            t_range = int(0.1 * Fs)

            if new_point - t_range < 0:
                new_point = np.argmax(np.abs(diff[:new_point + t_range]))

            elif new_point >= len(diff) - t_range:
                new_point = np.argmax(np.abs(diff[new_point - t_range:])) + new_point - t_range

            else:
                new_point = np.argmax(np.abs(diff[new_point - t_range:new_point + t_range])) + new_point - t_range

            temp_points.append(new_point)
            temp_points.append(new_point)

    for p in temp_points:
        neighbors = np.abs(new_ma_diff[p - step:p + step])
        if neighbors.size == 0:
            new_p = p
        else:
            new_p = np.argmax(neighbors) + (p - step)

        if new_p == 0:
            znak = 0
        else:
            znak = new_ma_diff[new_p] / np.abs([new_p])

        sacc_snake = new_ma_diff * znak
        sacc_snake[sacc_snake > 0] = 1
        sacc_snake[sacc_snake < 0] = -1

        s_i = 1
        while True:
            if p + s_i > sacc_snake.shape[0] - 1:
                break
            elif sacc_snake[p + s_i] >= 0:
                final_saccades.append(p + s_i)

            else:
                break

            s_i += 1

        s_i = 0
        while True or p - s_i > 0:
            if p + s_i == 0:
                break
            elif sacc_snake[p + s_i] >= 0:
                final_saccades.append(p + s_i)
            else:
                break
            s_i -= 1

    final_saccades.sort()
    final_saccades = list(set(final_saccades))
    final_saccades = np.array(final_saccades)

    diff_break_sacc = np.diff(final_saccades)
    where_break = np.where(diff_break_sacc > 1)[0]
    to_add = 0
    threshold = 30
    for i, pp in enumerate(where_break):
        bp = pp + to_add
        if diff_break_sacc[pp] <= threshold:
            temp_scc = final_saccades[bp + 1]
            j = 1
            while final_saccades[bp] + j < temp_scc:
                final_saccades = np.insert(final_saccades, bp + j, final_saccades[bp] + j)
                j += 1
                to_add += 1

    final_saccades = np.sort(final_saccades)

    _breaks = np.where(np.diff(final_saccades) > 1)[0]
    _to_little_breaks = np.where(np.diff(final_saccades) < 20)[0]
    to_fill = np.intersect1d(_breaks, _to_little_breaks)

    for i_f in range(0, to_fill.size // 2, 2):
        _to_add = 0
        for j_f in range(1, to_fill[i_f + 1] - to_fill[i_f]):
            final_saccades = np.insert(final_saccades, to_fill[i_f] + j_f, final_saccades[to_fill[i_f]] + j_f)

    return final_saccades
