import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import phenograph
import umap
import logging
import time
from datetime import datetime as dt
import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

from cellgan.lib.data_utils import load_fcs, get_fcs_filenames
from cellgan.lib.utils import compute_f_measure, build_logger
from cellgan.experiments.baselines.PhenoGraph.defaults import *


def main():
    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('--markers', nargs='+', type=str, default=DEFAULT_MARKERS,
                        help='List of markers')

    parser.add_argument('--in_dir', dest='input_dir', default=DATA_DIR,
                        help='Directory containing the input .fcs files')

    parser.add_argument('-o', '--out_dir', dest='output_dir', default=DEFAULT_OUT_DIR,
                        help='Directory where output will be generated.')

    # data processing
    parser.add_argument('--sub_limit', dest='subpopulation_limit', type=int, default=30,
                        help='Minimum number of cells to be called a subpopulation')

    parser.add_argument('--cofactor', dest='cofactor', type=int, default=5,
                        help='cofactor for the arcsinh transformation')

    parser.add_argument('--inhibitor', dest='inhibitor',
                        default=DEFAULT_INHIBITOR, help='Which inhibitor is used')

    parser.add_argument('--strength', dest='inhib_strength', default=DEFAULT_INHIB_STRENGTH,
                        help='Strength of inhibitor used')

    parser.add_argument('--n_runs', dest='n_runs', default=1, type=int,
                        help='Run PhenoGraph multiple times to evaluate the performance.')

    parser.add_argument('--n_f_measure_rare', default=250, type=int,
                        help='Compute f-measure for rare subpopulations up to n_f_measure_rare.')
    args = parser.parse_args()

    # Setup the output directory
    experiment_name = dt.now().strftime("%d_%m_%Y-%H_%M_%S")
    output_dir = os.path.join(args.output_dir, args.inhibitor, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    markers_of_interest = args.markers
    inhibitor = args.inhibitor
    inhibitor_strength = args.inhib_strength
    fcs_savefile = os.path.join(output_dir, 'fcs.csv')
    markers_savefile = os.path.join(output_dir, 'markers.csv')

    fcs_files_of_interest = get_fcs_filenames(args.input_dir, inhibitor, inhibitor_strength)

    # Saving list of files and markers to output directory
    with open(fcs_savefile, 'w') as f:
        f.write(json.dumps(fcs_files_of_interest))
    with open(markers_savefile, 'w') as f:
        f.write(json.dumps(markers_of_interest))

    # Build logger
    pheno_logger = build_logger(out_dir=output_dir, logging_format='%(message)s', level=logging.INFO)

    # Data Loading
    pheno_logger.info('Starting to load and process the .fcs files...')
    start_time = time.time()

    training_data, training_labels = load_fcs(
        fcs_files=fcs_files_of_interest,
        markers=markers_of_interest,
        args=args,
        logger=pheno_logger
    )

    training_data = np.vstack(training_data)
    training_labels = np.concatenate(training_labels)
    training_labels_unique = np.unique(training_labels)

    pheno_logger.info("Loading and processing completed.")
    pheno_logger.info(
        'TIMING: File loading and processing took {} seconds \n'.format(
            datetime.timedelta(seconds=time.time() - start_time)))

    # get indices for rare subpopulation
    ind_training_labels_rare = list()
    for lab in training_labels_unique:
        temp_ind = np.where(training_labels == lab)[0]
        if len(temp_ind) < args.n_f_measure_rare:
            ind_training_labels_rare.append(temp_ind)
    ind_training_labels_rare = np.concatenate(ind_training_labels_rare)

    training_labels_rare = np.zeros(training_labels.shape)
    training_labels_rare[ind_training_labels_rare] = training_labels[ind_training_labels_rare]

    # get marker sizes
    marker_sizes = list()
    for i, subpop in enumerate(training_labels_unique):
        temp_ind = np.where(training_labels == subpop)[0]
        marker_sizes.append(len(temp_ind))

    for i in range(len(marker_sizes)):
        marker_sizes[i] = -np.log(marker_sizes[i] / float(training_data.shape[0]))

    # Fit PCA object
    pca = PCA(n_components=2)
    pca_transform = pca.fit_transform(training_data)

    # Fit UMAP object
    um = umap.UMAP()
    um_transform = um.fit_transform(training_data)

    raise ValueError()

    # Fit TSNE object
    tsne = TSNE(n_components=2)
    tsne_transform = tsne.fit_transform(training_data)

    # apply phenograph
    list_communities = list()
    list_f_measure = list()
    list_f_measure_rare = list()
    for i in range(args.n_runs):
        communities, graph, Q = phenograph.cluster(training_data)
        list_communities.append(communities)
        f_measure = compute_f_measure(training_labels, communities)
        list_f_measure.append(f_measure)

        # set all communities zero which are not part of rare subpopulations
        communities_rare = np.zeros(communities.shape)
        communities_unique_rare = np.unique(communities[ind_training_labels_rare])
        for com in communities_unique_rare:
            temp_ind = np.where(communities == com)[0]
            communities_rare[temp_ind] = com

        f_measure_rare = compute_f_measure(training_labels_rare, communities_rare)
        list_f_measure_rare.append(f_measure_rare)


    plt.figure()
    plt.hist(list_f_measure, bins=20)
    plt.xlabel('F-measure')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hist_f-measures.pdf'))
    plt.close()

    pd.DataFrame(list_f_measure).to_csv(os.path.join(output_dir, 'f-measures.csv'), sep=',', header=False, index=False)


    plt.figure()
    plt.hist(list_f_measure_rare, bins=20)
    plt.xlabel('F-measure')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hist_f-measures_rare.pdf'))
    plt.close()

    pd.DataFrame(list_f_measure_rare).to_csv(os.path.join(output_dir, 'f-measures_rare.csv'), sep=',', header=False, index=False)

    ind_max = np.argmax(list_f_measure)
    communities = list_communities[ind_max]
    communities_unique = np.unique(communities)
    communities_unique_rare = np.unique(communities[ind_training_labels_rare])
    f_measure = list_f_measure[ind_max]
    f_measure_rare = list_f_measure_rare[ind_max]

    # get marker sizes
    marker_sizes_pG = list()
    for i, subpop in enumerate(communities_unique):
        temp_ind = np.where(communities == subpop)[0]
        marker_sizes_pG.append(len(temp_ind))

    for i in range(len(marker_sizes_pG)):
        marker_sizes_pG[i] = -np.log(marker_sizes_pG[i] / float(len(communities)))

    plt.figure()
    plt.subplot(121)
    cmap = matplotlib.cm.get_cmap('viridis')
    colors_truth = cmap(np.linspace(0, 1, len(training_labels_unique)))
    for i, subsets in enumerate(training_labels_unique):
        temp_ind = np.where(np.asarray(training_labels) == subsets)[0]
        plt.scatter(pca_transform[temp_ind, 0], pca_transform[temp_ind, 1], c=colors_truth[i], label=subsets, s=marker_sizes[i])
    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('ground truth')
    plt.tight_layout()

    plt.subplot(122)
    cmap = matplotlib.cm.get_cmap('viridis')
    colors_pg = cmap(np.linspace(0, 1, len(communities_unique)))
    for i, subsets in enumerate(communities_unique):
        temp_ind = np.where(np.asarray(communities) == subsets)[0]
        plt.scatter(pca_transform[temp_ind, 0], pca_transform[temp_ind, 1], c=colors_pg[i], label=subsets, s=marker_sizes_pG[i])
    plt.legend(loc='best')
    plt.title('PhenoGraph \n f-measure = ' + str(round(f_measure, 3)))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca.pdf'))
    plt.close()


    plt.figure()
    plt.subplot(121)
    for i, subsets in enumerate(training_labels_unique):
        temp_ind = np.where(np.asarray(training_labels) == subsets)[0]
        plt.scatter(um_transform[temp_ind, 0], um_transform[temp_ind, 1], c=colors_truth[i], label=subsets, s=marker_sizes[i])
    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('ground truth')
    plt.tight_layout()

    plt.subplot(122)
    for i, subsets in enumerate(communities_unique):
        temp_ind = np.where(np.asarray(communities) == subsets)[0]
        plt.scatter(um_transform[temp_ind, 0], um_transform[temp_ind, 1], c=colors_pg[i], label=subsets, s=marker_sizes_pG[i])
    plt.legend(loc='best')
    plt.title('PhenoGraph \n f-measure = ' + str(round(f_measure, 3)))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap.pdf'))
    plt.close()

    plt.figure()
    plt.subplot(121)
    for i, subsets in enumerate(training_labels_unique):
        temp_ind = np.where(np.asarray(training_labels) == subsets)[0]
        plt.scatter(tsne_transform[temp_ind, 0], tsne_transform[temp_ind, 1], c=colors_truth[i], label=subsets, s=marker_sizes[i])
    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('ground truth')
    plt.tight_layout()

    plt.subplot(122)
    for i, subsets in enumerate(communities_unique):
        temp_ind = np.where(np.asarray(communities) == subsets)[0]
        plt.scatter(tsne_transform[temp_ind, 0], tsne_transform[temp_ind, 1], c=colors_pg[i], label=subsets, s=marker_sizes_pG[i])
    plt.legend(loc='best')
    plt.title('PhenoGraph \n f-measure = ' + str(round(f_measure, 3)))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne.pdf'))
    plt.close()

    # rare subpopulations
    plt.figure()
    plt.subplot(121)
    cmap = matplotlib.cm.get_cmap('viridis')
    colors_truth = cmap(np.linspace(0, 1, len(training_labels_unique)))
    for i, subsets in enumerate(training_labels_unique):
        temp_ind = np.where(np.asarray(training_labels) == subsets)[0]
        if len(temp_ind) < args.n_f_measure_rare:
            plt.scatter(pca_transform[temp_ind, 0], pca_transform[temp_ind, 1], c=colors_truth[i], label=subsets, s=marker_sizes[i])
    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('ground truth')
    plt.tight_layout()

    plt.subplot(122)
    cmap = matplotlib.cm.get_cmap('viridis')
    colors_pg = cmap(np.linspace(0, 1, len(communities_unique_rare)))
    for i, subsets in enumerate(communities_unique_rare):
        temp_ind = np.where(np.asarray(communities) == subsets)[0]
        temp_ind = np.intersect1d(temp_ind, ind_training_labels_rare)
        plt.scatter(pca_transform[temp_ind, 0], pca_transform[temp_ind, 1], c=colors_pg[i], label=subsets, s=marker_sizes_pG[i])
    plt.legend(loc='best')
    plt.title('PhenoGraph \n f-measure = ' + str(round(f_measure_rare, 3)))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_rare.pdf'))
    plt.close()

    plt.figure()
    plt.subplot(121)
    for i, subsets in enumerate(training_labels_unique):
        temp_ind = np.where(np.asarray(training_labels) == subsets)[0]
        if len(temp_ind) < args.n_f_measure_rare:
            plt.scatter(um_transform[temp_ind, 0], um_transform[temp_ind, 1], c=colors_truth[i], label=subsets, s=marker_sizes[i])
    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('ground truth')
    plt.tight_layout()

    plt.subplot(122)
    for i, subsets in enumerate(communities_unique_rare):
        temp_ind = np.where(np.asarray(communities) == subsets)[0]
        temp_ind = np.intersect1d(temp_ind, ind_training_labels_rare)
        plt.scatter(um_transform[temp_ind, 0], um_transform[temp_ind, 1], c=colors_pg[i], label=subsets, s=marker_sizes_pG[i])
    plt.legend(loc='best')
    plt.title('PhenoGraph \n f-measure = ' + str(round(f_measure_rare, 3)))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_rare.pdf'))
    plt.close()

    plt.figure()
    plt.subplot(121)
    for i, subsets in enumerate(training_labels_unique):
        temp_ind = np.where(np.asarray(training_labels) == subsets)[0]
        if len(temp_ind) < args.n_f_measure_rare:
            plt.scatter(tsne_transform[temp_ind, 0], tsne_transform[temp_ind, 1], c=colors_truth[i], label=subsets, s=marker_sizes[i])
    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('ground truth')
    plt.tight_layout()

    plt.subplot(122)
    for i, subsets in enumerate(communities_unique_rare):
        temp_ind = np.where(np.asarray(communities) == subsets)[0]
        temp_ind = np.intersect1d(temp_ind, ind_training_labels_rare)
        plt.scatter(tsne_transform[temp_ind, 0], tsne_transform[temp_ind, 1], c=colors_pg[i], label=subsets, s=marker_sizes_pG[i])
    plt.legend(loc='best')
    plt.title('PhenoGraph \n f-measure = ' + str(round(f_measure_rare, 3)))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_rare.pdf'))
    plt.close()


if __name__ == '__main__':
    main()
