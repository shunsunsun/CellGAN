import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# TODO: Fix plotting.py

def pca_plot(out_dir, real_data, fake_data, it, real_subs, experts):

    subpops = np.unique(real_subs)
    markers = np.arange(real_data.shape[-1])

    ncol = markers.shape[0]
    nrow = len(subpops)

    dirname = os.path.join(out_dir, str((it//100)+1))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    unique_experts = np.unique(experts)

    for expert in unique_experts:

        filename = os.path.join(dirname, 'Expert_' + str(expert + 1) + '.png')
        expert_data = fake_data[np.where(experts == expert)[0], :]

        f, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(25, 25))
        best_sub = None
        best_KS_sum = np.inf

        for sub in subpops:

            KS_markers = list()

            indices = np.where(real_subs == sub)[0]
            sub_data = real_data[indices, :]

            for marker in markers:

                KS = ks_2samp(sub_data[:, marker], expert_data[:, marker])[0]
                KS_markers.append(KS)

                bins = np.linspace(sub_data[:, marker].min(), sub_data[:, marker].max(), num=30)
                ax[sub, marker].hist(x=sub_data[:, marker], bins=bins, label='R', alpha=0.5)
                ax[sub, marker].hist(x=expert_data[:, marker], bins=bins, label='F', alpha=0.5)
                ax[sub, marker].set_title('Sub {0}, Mark {1}, KS {2:.2f}'.format(sub + 1, marker + 1, KS))
                ax[sub, marker].legend()

            if np.sum(KS_markers) <= best_KS_sum:
                best_KS_sum = np.sum(KS_markers)
                best_sub = sub

        for marker in markers:

            ax[best_sub, marker].spines['bottom'].set_color('0.0')
            ax[best_sub, marker].spines['top'].set_color('0.0')
            ax[best_sub, marker].spines['right'].set_color('0.0')
            ax[best_sub, marker].spines['left'].set_color('0.0')
            [i.set_linewidth(2.5) for i in ax[best_sub, marker].spines.values()]

        f.tight_layout()
        plt.savefig(filename)
        plt.close()