from average_tf_map import average_tf
import numpy as np
import os
import scipy.stats as st
from mne.stats import fdr_correction


def stats(arr1, arr2):

    chosen_channels = ['Fp1', 'Fp2',
                       'F7', 'F3', 'Fz', 'F4', 'F8',
                       'T7', 'C3', 'Cz', 'C4', 'T8',
                       'P7', 'P3', 'Pz', 'P4', 'P8',
                       'O1', 'O2']

    for ch in range(arr1.shape[0]):

        print("Kanał ", chosen_channels[ch])

        #test nieparametryczny
        #https: // docs.scipy.org / doc / scipy / reference / generated / scipy.stats.wilcoxon.html
        z, p = st.wilcoxon(arr1[ch, :, :].flatten(), arr2[ch, :, :].flatten())
        p_w = p / 2  # aby test był jednostronny

        #kontrola frakcji fałszywych odkryć (FDR)
        # https://mne.tools/dev/auto_examples/stats/fdr_stats_evoked.html#sphx-glr-auto-examples-stats-fdr-stats-evoked-py
        reject_fdr, pval_fdr = fdr_correction(p, alpha=0.05, method='indep')


        s_w = u'prawdopodobieństwo dla nieparametrycznego testu jednostronnego wynosi %(p_h0).10f i %(p_fdr).10f po poprawce fdr' % {'p_h0': p_w, 'p_fdr': pval_fdr}

        print(s_w)

        print("Czy odrzucić hipotezę zerową: ", reject_fdr, "\n")

if __name__ == "__main__":

    filename = os.path.join("output", "data_matrix.npy")
    average = average_tf(filename)  # 19 kan, 2 serie, 60 x 500

    # uśrednianie po osobach
    avg1 = np.mean(average[:, :, 0, :, :], axis=0)
    avg2 = np.mean(average[:, :, 1, :, :], axis=0)

    #H0: średnie w obu seriach są takie same
    stats(avg1, avg2)

