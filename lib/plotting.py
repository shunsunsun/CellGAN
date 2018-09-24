import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def plot_marker_distributions(out_dir, real_subset, fake_subset, fake_subset_labels, real_subset_labels,
                              num_subpopulations, num_markers, num_experts, iteration, zero_sub=False, pca=True):

    """
    Plots the marker distribution per expert for each subpopulation and computes KS test and picks the best matching
    subpopulation for that expert
    :param out_dir: Output directory to save the plots to
    :param real_subset: Subset of real data
    :param fake_subset: Subset of fake data (generated by the GAN)
    :param fake_subset_labels: Which expert generated which data point
    :param real_subset_labels: Which subpopulation does the real data belong to
    :param num_subpopulations: Number of subpopulations in the training data
    :param num_markers: Number of markers whose distribution we tried to learn
    :param num_experts: Number of experts used in the generator
    :param iteration: iteration no.
    :param zero_sub: Whether the subpopulation labels start with zero or one
    :param pca: To add an additional plot with pca
    :return:
    """

    # TODO: Add the part for pca based plotting

    dirname = os.path.join(out_dir, str((iteration // 100) + 1))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for expert in range(num_experts):

        if pca:
            f, axes = plt.subplots(nrows=num_subpopulations, ncols=num_markers + 1, figsize=(30, 30))
        else:
            f, axes = plt.subplots(nrows=num_subpopulations, ncols=num_markers, figsize=(30, 30))

        best_ks_sum = np.inf

        filename = os.path.join(dirname, 'Expert_' + str(expert + 1) + '.png')

        indices = np.where(fake_subset_labels == expert)[0]

        # Fake data generated by expert in the GAN
        fake_data_by_expert = fake_subset[indices, :]

        for sub in range(num_subpopulations):

            if zero_sub:
                indices = np.where(real_subset_labels == sub)[0]
            else:
                indices = np.where(real_subset_labels == (sub + 1))[0]

            real_data_by_sub = real_subset[indices, :]

            ks_markers = list()

            for marker in range(num_markers):

                fake_max = np.max(fake_data_by_expert[:, marker])
                fake_min = np.min(fake_data_by_expert[:, marker])

                real_max = np.max(real_data_by_sub[:, marker])
                real_min = np.min(real_data_by_sub[:, marker])

                overall_max = max(real_max, fake_max)
                overall_min = min(real_min, fake_min)

                bins = np.linspace(overall_min, overall_max, num=30)

                w = np.ones_like(real_data_by_sub)/float(len(real_data_by_sub))
                axes[sub, marker].hist(real_data_by_sub, bins=bins, weights=w, label='R', normed=0, alpha=0.5)

                w = np.ones_like(fake_data_by_expert)/float(len(fake_data_by_expert))
                axes[sub, marker].hist(fake_data_by_expert, bins=bins, weights=w, label='F', normed=0, alpha=0.5)

                ks = ks_2samp(fake_data_by_expert[:, marker], real_data_by_sub[:, marker])[0]
                ks_markers.append(ks)

                axes[sub, marker].set_xlim([overall_min, overall_max])
                ticks = np.linspace(overall_min, overall_max, num=5)
                axes[sub, marker].set_xticks(ticks.round(2))

                axes[sub, marker].set_title('Marker {}'.format(marker))
                axes[sub, marker].set_ylabel('Subpopulation {}'.format(sub))

            if np.sum(ks_markers) < best_ks_sum:
                best_ks_sum = np.sum(ks_markers)
                best_sub = sub

        for marker in range(num_markers):
            axes[best_sub, marker].spines['bottom'].set_color('0.0')
            axes[best_sub, marker].spines['top'].set_color('0.0')
            axes[best_sub, marker].spines['right'].set_color('0.0')
            axes[best_sub, marker].spines['left'].set_color('0.0')
            [i.set_linewidth(2.5) for i in axes[best_sub, marker].spines.values()]

        f.suptitle('Marker Distribution Plots per subpopulation', x=0.5, y=1.02, fontsize=20)
        f.tight_layout()
        plt.savefig(filename)
        plt.close()

        print('Marker distribution plot for expert {} added.'.format(expert+1))
