import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

from cellgan.lib.utils import compute_l2


def plotter(out_dir, method, transformer, real_subset, real_subset_labels, 
            fake_subset, fake_subset_labels, num_experts, num_subpopulations, iteration, logger, 
            zero_sub=False, all_real_vs_expert=False, each_subpop=False):

    """ Generates plots for each expert based on given method 
    :param out_dir: results directory
    :param method: What type of plot (one of {pca, umap, tsne})
    :param transformer: An object with transform method accompanying the method
    :param real_subset: The sampled subset of real data
    :param real_subset_labels: Subpopulations associated with each cell in real subset
    :param fake_subset: Fake data from the generator 
    :param fake_subset_labels: Which expert generates which cell in fake data
    :param num_experts: Number of experts used
    :param num_subpopulations: Number of subpopulations in the data
    :param iteration: which iteration of training are we at
    :param logger: logger used
    :param zero_sub: Whether subpopulations labelled from 0 or 1
    :param all_real_vs_expert: Whether to plot each expert vs real subset
    :param each_subpop: Whether to plot each expert vs each subpopulation

    """
    # The actual directory where results are saved
    dirname = os.path.join(out_dir, str(iteration + 1))
    save_dir = os.path.join(dirname, method + '_plots')

    # Transform data according to transformer based on method
    transformed = transformer.transform(np.vstack([real_subset, fake_subset]))
    transformed_real = transformed[:real_subset.shape[0], :]
    transformed_fake = transformed[real_subset.shape[0]:, :]

    label_dict = {'pca': ['PC1', 'PC2'], 'umap': ['UM1', 'UM2'], 'tsne': ['TSNE1', 'TSNE2']}
    labels = label_dict[method]

    # All real vs all expert
    plt.figure()
    cmap = matplotlib.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(np.unique(real_subset_labels))))
    for i, subsets in enumerate(np.unique(real_subset_labels)):
        temp_ind = np.where(np.asarray(real_subset_labels) == subsets)[0]
        plt.scatter(transformed_real[temp_ind, 0], transformed_real[temp_ind, 1], c=colors[i].reshape((1, 4)), s=1)
    plt.scatter(transformed_fake[:, 0], transformed_fake[:, 1], c='red', s=1)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, method + '_all-real_vs_all_expert.pdf'))
    plt.close()

    for expert in range(num_experts):
        indices = np.flatnonzero(fake_subset_labels == expert)
        fake_data_by_expert = transformed_fake[indices]

        if each_subpop:
            expert_dir = os.path.join(save_dir, 'Expert_{}'.format(expert+1))
            if not os.path.exists(expert_dir):
                os.makedirs(expert_dir)

            plot_subpop_vs_expert(method=method, expert_dir=expert_dir, expert=expert,
                                  transformed_real=transformed_real, real_subset_labels=real_subset_labels,
                                  fake_data_by_expert=fake_data_by_expert, num_subpopulations=num_subpopulations,
                                  zero_sub=zero_sub)

        if all_real_vs_expert:
            expert_dir = os.path.join(save_dir, 'Expert_{}'.format(expert+1))
            if not os.path.exists(expert_dir):
                os.makedirs(expert_dir)
                
            plot_all_real_vs_expert(out_dir=out_dir, method=method, transformed_real=transformed_real,
                                    real_subset_labels=real_subset_labels, fake_data_by_expert=fake_data_by_expert,
                                    expert=expert, iteration=iteration)

    if logger is not None:
        logger.info("\n")


def plot_subpop_vs_expert(method, expert_dir, expert, transformed_real, real_subset_labels,
                          fake_data_by_expert, num_subpopulations, zero_sub=False):
    """Generates plot for expert vs each subpopulation for given method."""

    label_dict = {'pca': ['PC1', 'PC2'], 'umap': ['UM1', 'UM2'], 'tsne': ['TSNE1', 'TSNE2']}
    labels = label_dict[method]

    for subpopulation in range(num_subpopulations):
        if zero_sub:
            indices = np.flatnonzero(real_subset_labels == subpopulation)
        else:
            indices = np.flatnonzero(real_subset_labels == subpopulation + 1)
        real_data_by_sub = transformed_real[indices]

        f, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

        # (Real Subpopulation, Fake Data)
        axes[1].scatter(real_data_by_sub[:, 0], real_data_by_sub[:, 1], c='tab:blue',
                        label='Subpopulation {}'.format(subpopulation + 1))
        axes[1].scatter(fake_data_by_expert[:, 0], fake_data_by_expert[:, 1], c='tab:orange',
                        label='Expert {}'.format(expert + 1))
        axes[1].legend()

        # Real subpopulation
        xmin, xmax = axes[1].get_xlim()
        ymin, ymax = axes[1].get_ylim()

        axes[0].scatter(real_data_by_sub[:, 0], real_data_by_sub[:, 1], c='tab:gray',
                        label='Subpopulation {}'.format(subpopulation + 1))
        axes[0].set_xlim([xmin, xmax])
        axes[0].set_ylim([ymin, ymax])
        axes[0].legend()

        axes[1].set_xlabel(labels[0])
        axes[1].set_ylabel(labels[1])
        axes[0].set_xlabel(labels[0])
        axes[0].set_ylabel(labels[1])

        savefile = os.path.join(expert_dir, 'Subpopulation_{}.png'.format(subpopulation + 1))
        f.tight_layout()
        plt.savefig(savefile)
        plt.close()


def plot_all_real_vs_expert(out_dir, method, transformed_real, real_subset_labels,
                            fake_data_by_expert, expert, iteration):
    """Generates plot of expert vs real subpopulations."""

    dirname = os.path.join(out_dir, str(iteration + 1))
    save_dir = os.path.join(dirname, method + '_plots')

    label_dict = {'pca': ['PC1', 'PC2'], 'umap': ['UM1', 'UM2'], 'tsne': ['TSNE1', 'TSNE2']}
    labels = label_dict[method]

    plt.figure()
    cmap = matplotlib.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(np.unique(real_subset_labels))))
    for i, subsets in enumerate(np.unique(real_subset_labels)):
        temp_ind = np.where(np.asarray(real_subset_labels) == subsets)[0]
        plt.scatter(transformed_real[temp_ind, 0], transformed_real[temp_ind, 1], c=colors[i].reshape((1, 4)),
                    s=1)
    plt.scatter(fake_data_by_expert[:, 0], fake_data_by_expert[:, 1], c='red', s=1)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Expert {}'.format(expert+1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'All-real_vs_expert_' + str(expert+1) + '.png'))
    plt.close()


def plot_pca(out_dir, pca_obj, real_subset, real_subset_labels, fake_subset,
             fake_subset_labels, num_experts, num_subpopulations, iteration, logger,
             zero_sub=False, all_real_vs_expert=False, each_subpop=False):
    """Generates the PCA plot for each expert."""

    plotter(out_dir=out_dir, method='pca', transformer=pca_obj, 
            real_subset=real_subset, real_subset_labels=real_subset_labels, 
            fake_subset=fake_subset, fake_subset_labels=fake_subset_labels, 
            num_experts=num_experts, num_subpopulations=num_subpopulations, 
            iteration=iteration, logger=logger, zero_sub=zero_sub,
            all_real_vs_expert=all_real_vs_expert, each_subpop=each_subpop)


def plot_umap(out_dir, umap_obj, real_subset, real_subset_labels, fake_subset,
              fake_subset_labels, num_experts, num_subpopulations, iteration, logger,
              zero_sub=False, all_real_vs_expert=False, each_subpop=False):
    """Generates the UMAP plot."""
    
    plotter(out_dir=out_dir, method='umap', transformer=umap_obj, 
            real_subset=real_subset, real_subset_labels=real_subset_labels, 
            fake_subset=fake_subset, fake_subset_labels=fake_subset_labels, 
            num_experts=num_experts, num_subpopulations=num_subpopulations, 
            iteration=iteration, logger=logger, zero_sub=zero_sub,
            all_real_vs_expert=all_real_vs_expert, each_subpop=each_subpop)


def plot_tsne(out_dir, tsne_obj, real_subset, real_subset_labels, fake_subset,
              fake_subset_labels, num_experts, num_subpopulations, iteration, logger,
              zero_sub=False, all_real_vs_expert=False, each_subpop=False):
    """ Generates the tsne plot """

    plotter(out_dir=out_dir, method='tsne', transformer=tsne_obj, 
            real_subset=real_subset, real_subset_labels=real_subset_labels, 
            fake_subset=fake_subset, fake_subset_labels=fake_subset_labels, 
            num_experts=num_experts, num_subpopulations=num_subpopulations,
            iteration=iteration, logger=logger, zero_sub=zero_sub,
            all_real_vs_expert=all_real_vs_expert, each_subpop=each_subpop)


def plot_marker_distributions(out_dir,
                              real_subset,
                              fake_subset,
                              fake_subset_labels,
                              real_subset_labels,
                              num_subpopulations,
                              num_markers,
                              num_experts,
                              marker_names,
                              iteration,
                              logger=None,
                              zero_sub=False):
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
    :param marker_names: Names of markers
    :param iteration: iteration no.
    :param logger: logger used for logging results
    :param zero_sub: Whether the subpopulation labels start with zero or one
    :param pca: To add an additional plot with pca
    :return:
    """
    dirname = os.path.join(out_dir, str(iteration + 1))
    save_dir = os.path.join(dirname, 'distributions')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for expert in range(num_experts):
        f, axes = plt.subplots(nrows=num_subpopulations, ncols=num_markers, figsize=(30, 30))
        best_ks_sum = np.inf

        # Fake data generated by expert in the GAN
        filename = os.path.join(save_dir, 'Expert_' + str(expert + 1) + '.png')
        indices = np.flatnonzero(fake_subset_labels == expert)
        fake_data_by_expert = fake_subset[indices, :]

        for sub in range(num_subpopulations):
            if zero_sub:
                indices = np.flatnonzero(real_subset_labels == sub)
            else:
                indices = np.flatnonzero(real_subset_labels == (sub + 1))

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

                w = np.ones_like(real_data_by_sub[:, marker]) / float(len(real_data_by_sub[:, marker]))
                axes[sub, marker].hist(real_data_by_sub[:, marker], bins=bins,
                                       weights=w, label='R', normed=0, alpha=0.5)

                w = np.ones_like(fake_data_by_expert[:, marker]) / float(len(fake_data_by_expert[:, marker]))
                axes[sub, marker].hist(fake_data_by_expert[:, marker], bins=bins,
                                       weights=w, label='F', normed=0, alpha=0.5)

                ks = ks_2samp(fake_data_by_expert[:, marker],
                              real_data_by_sub[:, marker])[0]
                ks_markers.append(ks)

                axes[sub, marker].set_xlim([overall_min, overall_max])
                ticks = np.linspace(overall_min, overall_max, num=5)
                axes[sub, marker].set_xticks(ticks.round(2))

                axes[sub, marker].set_title(marker_names[marker] + ', KS:' + str(np.round(ks, 3)))
                axes[sub, marker].set_ylabel(
                    'Subpopulation {}'.format(sub + 1))
                axes[sub, marker].legend()

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

        if logger is not None:
            logger.info('Marker distribution plot for expert {} added.'.format(expert + 1))


def plot_heatmap(out_dir, iteration, logits, fake_subset_labels):

    """
    Heat map plot
    :param out_dir: str, output directory
    :param iteration: iteration number
    :param logits: Tensor, unnormalized log probs of an expert generating a cell
    :param fake_subset_labels: Which expert generates which cell
    """
    filename = os.path.join(out_dir, str(iteration+1), 'heatmap.pdf')
    unique_experts = np.unique(fake_subset_labels)
    expert_labels_series = pd.Series(fake_subset_labels)
    lut = dict(zip(expert_labels_series.unique(),
                   plt.cm.get_cmap('jet', len(unique_experts))(np.linspace(0, 1, len(unique_experts)))))

    row_colors = expert_labels_series.map(lut)
    plt.figure()
    g = sns.clustermap(logits, row_colors=list(row_colors),
                       yticklabels=False, xticklabels=True)
    for expert in unique_experts:
        g.ax_row_dendrogram.bar(0, 0, color=lut[expert], label=expert, linewidth=0)

    g.ax_row_dendrogram.legend(loc="best", ncol=1, fancybox=True, framealpha=0.5)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)  # For x axis
    plt.rcParams["xtick.labelsize"] = 6.5
    plt.savefig(filename)
    plt.close()


def plot_loss(out_dir, disc_loss, gen_loss):
    """
    Saves loss plot to output directory
    :param out_dir: str, output directory
    :param disc_loss: list, discriminator losses
    :param gen_loss: list, generator losses
    :return: no returns
    """
    filename = os.path.join(out_dir, 'loss_plot.pdf')
    plt.plot(range(len(disc_loss)), disc_loss, 'r', label='Discriminator Loss')
    plt.plot(range(len(gen_loss)), gen_loss, 'b', label='Generator Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_convergence(out_dir, iters, error):
    filename = os.path.join(out_dir, 'convergence_plot.pdf')
    plt.plot(iters, error, label='Reconstruction Error')
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_expert_vs_expert_markers(out_dir,
                                  fake_subset,
                                  fake_subset_labels,
                                  num_markers,
                                  num_experts,
                                  marker_names,
                                  zero_sub=False):
    """ Plots distribution of one-expert against all others"""

    save_dir = os.path.join(out_dir, 'expert-vs-expert')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for expert in range(num_experts):
        f, axes = plt.subplots(nrows=num_experts-1, ncols=num_markers, figsize=(30, 30))
        best_ks_sum = np.inf

        filename = os.path.join(save_dir, 'Expert_' + str(expert + 1) + '.pdf')
        indices = np.flatnonzero(fake_subset_labels == expert)

        # Fake data generated by expert in the GAN
        fake_data_by_expert = fake_subset[indices, :]
        experts_to_check = list(range(num_experts))
        experts_to_check.pop(expert)

        for i, other_expert in enumerate(experts_to_check):
            if other_expert == expert:
                continue

            if zero_sub:
                indices = np.flatnonzero(fake_subset_labels == other_expert)
            else:
                indices = np.flatnonzero(fake_subset_labels == (other_expert + 1))
            fake_data_by_other_expert = fake_subset[indices, :]
            ks_markers = list()

            for marker in range(num_markers):
                fake_max = np.max(fake_data_by_expert[:, marker])
                fake_min = np.min(fake_data_by_expert[:, marker])

                fake_other_max = np.max(fake_data_by_other_expert[:, marker])
                fake_other_min = np.min(fake_data_by_other_expert[:, marker])

                overall_max = max(fake_other_max, fake_max)
                overall_min = min(fake_other_min, fake_min)
                bins = np.linspace(overall_min, overall_max, num=30)

                w = np.ones_like(fake_data_by_other_expert[:, marker]) / float(len(fake_data_by_other_expert[:, marker]))
                axes[i, marker].hist(fake_data_by_other_expert[:, marker], bins=bins,
                                     weights=w, label='R', normed=0, alpha=0.5)

                w = np.ones_like(fake_data_by_expert[:, marker]) / float(len(fake_data_by_expert[:, marker]))
                axes[i, marker].hist(fake_data_by_expert[:, marker], bins=bins,
                                     weights=w, label='F', normed=0, alpha=0.5)

                ks = ks_2samp(fake_data_by_expert[:, marker],
                              fake_data_by_other_expert[:, marker])[0]
                ks_markers.append(ks)

                axes[i, marker].set_xlim([overall_min, overall_max])
                ticks = np.linspace(overall_min, overall_max, num=5)
                axes[i, marker].set_xticks(ticks.round(2))

                axes[i, marker].set_title('{}'.format(marker_names[marker]))
                axes[i, marker].set_ylabel(
                    'Expert {}'.format(other_expert + 1))
                axes[i, marker].legend()

            if np.sum(ks_markers) < best_ks_sum:
                best_ks_sum = np.sum(ks_markers)
                best_sub = i

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
