from average_tf_map import average_tf
import numpy as np
import os
import scipy.stats as st
from mne.stats import fdr_correction


def stats(arr1):

    chosen_channels = ['Fp1', 'Fp2',
                       'F7', 'F3', 'Fz', 'F4', 'F8',
                       'T7', 'C3', 'Cz', 'C4', 'T8',
                       'P7', 'P3', 'Pz', 'P4', 'P8',
                       'O1', 'O2']
    p_all = np.zeros((arr1.shape[1], arr1.shape[-2], arr1.shape[-1]))
    for ch in range(arr1.shape[1]):

        print("Kanał ", chosen_channels[ch])

        #test nieparametryczny
        #https: // docs.scipy.org / doc / scipy / reference / generated / scipy.stats.wilcoxon.html
        for f in range(arr1.shape[-2]):
            for t in range(arr1.shape[-1]):
                z, p_all[ch, f, t] = st.wilcoxon(average[:,ch, 0, f, t].flatten(), average[:,ch,1, f, t].flatten())
        #jest gdzieś różnica
        # test prostokąt po prostokącie w danym okienku (oddzielnie dla kanału)
        # dla osób (po osobach)
        # p (kanał x czas x częstość)!
        # p_w = p / 2  # aby test był jednostronny
                p_all[ch, f, t] = p_all[ch, f, t]/2

        #kontrola frakcji fałszywych odkryć (FDR)
        # https://mne.tools/dev/auto_examples/stats/fdr_stats_evoked.html#sphx-glr-auto-examples-stats-fdr-stats-evoked-py
    reject_fdr, pval_fdr = fdr_correction(p_all, alpha=0.05, method='indep') #można macierz
        # sprawdzić czy nie mniejsze do 1/2

        # tam gdzie true to nas interesuje reject_fdr pokazać w pełni reszte zamaskować

    # s_w = u'prawdopodobieństwo dla nieparametrycznego testu jednostronnego wynosi %(p_h0).10f i %(p_fdr).10f po poprawce fdr' % {'p_h0': p_w, 'p_fdr': pval_fdr}

    # print(s_w)

    # print("Czy odrzucić hipotezę zerową: ", reject_fdr, "\n")
    print(p_all)
    print(p_all.shape)
    print(reject_fdr)
    print(reject_fdr.shape)
    return p_all, reject_fdr

if __name__ == "__main__":

    filename = os.path.join("output", "data_matrix.npy")
    average = average_tf(filename)  # 19 kan, 2 serie, 60 x 500
    print(average.shape)
    # uśrednianie po osobach
    # avg1 = np.mean(average[:, :, 0, :, :], axis=0)
    # avg2 = np.mean(average[:, :, 1, :, :], axis=0)

    #H0: średnie w obu seriach są takie same
    stats(average)