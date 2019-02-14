""" GMM on Bodenmiller data as a baseline to CellGAN. """

import argparse
import json
import time
import logging
from datetime import datetime as dt
import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import umap

from cellgan.lib.preprocessing import extract_marker_indices, read_fcs_data
from cellgan.lib.utils import write_hparams_to_file, f_trans
from cellgan.lib.utils import build_logger, generate_subset
from cellgan.lib.utils import compute_frequency, assign_expert_to_subpopulation, compute_learnt_subpopulation_weights
from cellgan.lib.plotting import plot_marker_distributions, plot_pca, plot_umap
from cellgan.experiments.baselines.GMM.defaults import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('-f', '--fcs', dest='fcs_file', default=DEFAULT_FCS_FILE,
                        help='file containing names of .fcs files to be used for GAN training')

    parser.add_argument('-m', '--markers', dest='marker_file', default=DEFAULT_MARKERS_FILE,
                        help='Filename containing the markers of interest')

    parser.add_argument('-i', '--in_dir', dest='input_dir', default=DEFAULT_INPUT_DIR,
                        help='Directory containing the input .fcs files')

    parser.add_argument('-o', '--out_dir', dest='output_dir', default=DEFAULT_OUT_DIR,
                        help='Directory where output will be generated.')

    # data processing

    parser.add_argument('--sub_limit', dest='subpopulation_limit', type=int, default=30,
                        help='Minimum number of cells to be called a subpopulation')

    parser.add_argument('--cofactor', dest='cofactor', type=int, default=5,
                        help='cofactor for the arcsinh transformation')

    # GMM Specific

    parser.add_argument('-e', '--experts', dest='num_experts', type=int,
                        help='Number of experts in the generator')

    parser.add_argument('--cov_type', dest='cov_type', default='full',
                        choices=['full', 'tied', 'diag', 'spherical'],
                        help='Covariance types - For more details, see sklearn documentation')

    parser.add_argument('--max_iter', dest='max_iter', default=1000, type=int,
                        help='Number of EM iterations to perform using the GMM')

    parser.add_argument('--tol', dest='tol', default=1e-3, type=float,
                        help='Tolerance before convergence')

    parser.add_argument('--reg_covar', dest='reg_covar', default=1e-6, type=float,
                        help='Regularization added to covariance to ensure positive')

    parser.add_argument('--n_init', dest='n_init', default=1,
                        help='Number of initializations to perform')

    parser.add_argument('--init_method', dest='init_method', default='kmeans',
                        choices=['kmeans', 'random'], help='Initializing GMM parameters')

    # Testing specific

    parser.add_argument('--num_samples', dest='num_samples', type=int,
                        help='Number of samples to generate while testing')
    
    args = parser.parse_args()
    
    with open(args.fcs_file, 'r') as f:
        fcs_files_of_interest = json.load(f)

    with open(args.marker_file, 'r') as f:
        markers_of_interest = json.load(f)

    inhibitor_used = fcs_files_of_interest[0].split('_')[0]
    inhibitor_strength_used = fcs_files_of_interest[0].split('.')[0][-3:]

    # Setup the output directory
    experiment_name = dt.now().strftime("%d_%m_%Y-%H_%M_%S")
    output_dir = os.path.join(args.output_dir, experiment_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build logger
    gmm_logger = build_logger(
        out_dir=output_dir, logging_format='%(message)s', level=logging.INFO)

    # Data Loading Steps
    # ------------------

    gmm_logger.info('Starting to load and process the .fcs files...')
    start_time = time.time()

    training_data = list()
    training_labels = list(
    )  # These are not just for training, just for checking later
    celltype_added = 0

    # TODO: Replace this with .pkl file
    # TODO: Number of test samples == Number of training samples. Is that okay?

    for file in fcs_files_of_interest:

        file_path = os.path.join(args.input_dir, file.strip())
        fcs_data = read_fcs_data(file_path=file_path)

        try:
            marker_indices = extract_marker_indices(
                fcs_data=fcs_data, markers_of_interest=markers_of_interest)
            num_cells_in_file = fcs_data.data.shape[0]

            if num_cells_in_file >= args.subpopulation_limit:

                processed_data = np.squeeze(fcs_data.data[:, marker_indices])
                processed_data = f_trans(processed_data, c=args.cofactor)

                training_labels.append([celltype_added] * num_cells_in_file)
                celltype_added += 1

                training_data.append(processed_data)
                gmm_logger.info(
                    'File {} loaded and processed'.format(file))
                gmm_logger.info('File {} contains {} cells \n'.format(
                    file, num_cells_in_file))

            else:
                continue

        except AttributeError:
            pass

    gmm_logger.info("Loading and processing completed.")
    gmm_logger.info(
        'TIMING: File loading and processing took {} seconds \n'.format(
            datetime.timedelta(seconds=time.time() - start_time)))

    training_data = np.vstack(training_data)
    training_labels = np.concatenate(training_labels)

    # Actual subpopulation weights
    weights_subpopulations = compute_frequency(
        labels=training_labels, weighted=True)

    num_subpopulations = len(np.unique(training_labels))

    if not args.num_experts or args.num_experts <= num_subpopulations:
        num_experts = num_subpopulations
    else:
        num_experts = args.num_experts

    if not args.num_samples:
        num_samples = training_data.shape[0]
    else:
        num_samples = args.num_samples

    # Build GMM

    gmm_logger.info('Building GMM...')

    model = GaussianMixture(
        n_components=num_experts,
        covariance_type=args.cov_type,
        max_iter=args.max_iter,
        n_init=args.n_init,
        tol=args.tol,
        reg_covar=args.reg_covar,
        init_params=args.init_method
    )

    gmm_logger.info('GMM built. \n')

    model_hparams = {
        'n_components': num_experts,
        'cov_type': args.cov_type,
        'max_iter': args.max_iter,
        'n_init': args.n_init,
        'tol': args.tol,
        'reg_covar': args.reg_covar,
        'init_method': args.init_method,
    }

    # Save Hyperparameters used
    gmm_logger.info('Saving hyperparameters...')
    write_hparams_to_file(out_dir=output_dir, hparams=model_hparams)
    gmm_logger.info('Hyperparameters saved.')

    # Log data to output file
    gmm_logger.info("Experiment Name: " + experiment_name)
    gmm_logger.info("Inhibitor Used {} with strength {} ".format(
        inhibitor_used, inhibitor_strength_used))
    gmm_logger.info("Starting our baseline experiments with {} "
                    "subpopulations".format(num_subpopulations))

    # Fit PCA object
    pca = PCA(n_components=2)
    pca = pca.fit(training_data)

    # Fit UMAP object
    um = umap.UMAP(n_neighbors=10)
    um = um.fit(training_data)

    # Train the GMM

    model.fit(X=training_data, y=training_labels)

    # Test Model and Generate Plots

    fake_samples, fake_sample_experts = model.sample(num_samples)

    fake_samples = fake_samples.reshape(num_samples,
                                        len(markers_of_interest))

    # Sample real data for testing
    real_samples, indices = generate_subset(
        inputs=training_data,
        num_cells_per_input=num_samples,
        weights=None,  # Should I add weights differently?
        batch_size=1,
        return_indices=True)
    real_samples = real_samples.reshape(num_samples,
                                        len(markers_of_interest))
    indices = np.reshape(indices, real_samples.shape[0])
    real_sample_subs = training_labels[indices]

    # Compute expert assignments based on KS test
    expert_assignments = \
        assign_expert_to_subpopulation(real_data=real_samples, real_labels=real_sample_subs,
                                       fake_data=fake_samples, expert_labels=fake_sample_experts,
                                       num_experts=num_experts, num_subpopulations=num_subpopulations)

    # Compute learnt subpopulation weights
    learnt_subpopulation_weights = \
        compute_learnt_subpopulation_weights(expert_labels=fake_sample_experts,
                                             expert_assignments=expert_assignments,
                                             num_subpopulations=num_subpopulations)

    gmm_logger.info("The actual subpopulation weights are: {}".
                        format(weights_subpopulations))
    gmm_logger.info(
        "The learnt subpopulation weights are: {} \n".format(learnt_subpopulation_weights))

    # Plot marker distributions
    gmm_logger.info("Adding marker distribution plots...")

    plot_marker_distributions(
        out_dir=output_dir,
        real_subset=real_samples,
        fake_subset=fake_samples,
        real_subset_labels=real_sample_subs,
        fake_subset_labels=fake_sample_experts,
        num_experts=num_experts,
        num_markers=len(markers_of_interest),
        num_subpopulations=num_subpopulations,
        marker_names=markers_of_interest,
        iteration=0,
        logger=gmm_logger,
        zero_sub=True)

    gmm_logger.info("Marker distribution plots added. \n")

    # PCA plot
    gmm_logger.info("Adding PCA plots...")

    plot_pca(
        out_dir=output_dir, pca_obj=pca,
        real_subset=real_samples,
        fake_subset=fake_samples,
        real_subset_labels=real_sample_subs,
        fake_subset_labels=fake_sample_experts,
        num_experts=num_experts,
        num_subpopulations=num_subpopulations,
        iteration=0,
        logger=gmm_logger,
        zero_sub=True)

    gmm_logger.info("PCA plots added. \n")

    # UMAP plot
    gmm_logger.info("Adding UMAP plots...")

    plot_umap(
        out_dir=output_dir, umap_obj=um,
        real_subset=real_samples,
        fake_subset=fake_samples,
        real_subset_labels=real_sample_subs,
        fake_subset_labels=fake_sample_experts,
        num_experts=num_experts,
        num_subpopulations=num_subpopulations,
        iteration=0,
        logger=gmm_logger,
        zero_sub=True)

    gmm_logger.info("UMAP plots added. \n")


if __name__ == '__main__':

    main()



















