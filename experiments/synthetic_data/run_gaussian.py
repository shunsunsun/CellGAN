import argparse
import os
import sys
import time
import logging
from datetime import datetime as dt
import datetime

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import umap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

from lib.utils import get_filters, get_num_pooled, write_hparams_to_file
from lib.utils import generate_subset, sample_z, compute_outlier_weights
from lib.utils import build_logger, build_gaussian_training_set
from lib.utils import compute_frequency, assign_expert_to_subpopulation, compute_learnt_subpopulation_weights
from lib.model import CellGan
from lib.plotting import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/synthetic_data/')
DEFAULT_MARKERS = ['CD3', 'CD45', 'CD4', 'CD20', 'CD33', 'CD123', 'CD14', 'IgM', 'HLA-DR', 'CD7']


def main():

    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('-o', '--out_dir', dest='output_dir', default=DEFAULT_OUT_DIR,
                        help='Directory where output will be generated.')

    # Synthetic Data Specific
    parser.add_argument('--num_cells', default=40000, type=int,
                        help='Number of cells in the training set')

    parser.add_argument('--num_subpopulations', default=10, type=int,
                        help='Number of subpopulations on the training set')

    parser.add_argument('--num_markers', default=10, type=int,
                        help='Number of markers to simulate Gaussian data for')

    parser.add_argument('-w', '--weights_subpopulations', nargs='+', type=float, default=[0.1]*10,
                        help='Subpopulation weights')

    # multi-cell input specific
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=64,
                        help='batch size used for training')

    parser.add_argument('-nc', '--ncell', dest='num_cells_per_input', type=int, default=100,
                        help='Number of cells per multi-cell input')

    # Generator Parameters
    parser.add_argument('-ns', '--noise_size', dest='noise_size', type=int, default=100,
                        help='Noise dimension for generator')

    parser.add_argument('--moe_sizes', dest='moe_sizes', nargs='+', type=int, default=[100, 100],
                        help='Sizes of the Mixture of Experts hidden layers')

    parser.add_argument('-e', '--experts', dest='num_experts', type=int,
                        help='Number of experts in the generator')

    parser.add_argument('-g', '--g_filters', dest='num_filters_generator', type=int, default=20,
                        help='Number of filters in conv layer of generator')

    parser.add_argument('--n_top', dest='num_top', default=1, type=int,
                        help='Number of experts used for generating each cell')

    parser.add_argument('--noisy_gating', default=True,
                        help='Whether to add noise to gating weights at train time')

    parser.add_argument('--noise_eps', default=1e-2, type=float, help='Noise threshold')

    parser.add_argument('--load_balancing', action='store_true',
                        help='Whether to add load balancing to Mixture of experts')

    parser.add_argument('--moe_loss_coef', default=1e-1, type=float,
                        help='Loss coefficient for mixture of experts loss')

    # Discriminator Parameters

    parser.add_argument('--num_cell_cnns', default=30, type=int,
                        help='Number of CellCnns in the ensemble')

    parser.add_argument('--d_filters_min', default=7, type=int,
                        help='Minimum number of filters for a CellCnn')

    parser.add_argument('--d_filters_max', default=10, type=int,
                        help='Maximum number of filters for a CellCnn')

    parser.add_argument('-cl1', '--coeff_l1', dest='coeff_l1', default=0,
                        type=float, help='Coefficient for l1 regularizer')

    parser.add_argument('-cl2', '--coeff_l2', dest='coeff_l2', default=1e-4,
                        type=float, help='Coefficient for l2 regularizer')

    parser.add_argument('--coeff_act', dest='coeff_act', default=0,
                        type=float, help='No clue what this is')

    parser.add_argument('--dropout', dest='dropout_prob', type=float, default=0.5,
                        help='Dropout probability')

    # Training Specific

    parser.add_argument('--subset', dest='subset_sample', default='random',
                        choices=['random', 'outlier'],
                        help='Whether to bias the subset selection towards rare subpopulations')

    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float,
                        default=2e-4, help='Learning rate for the neural network')

    parser.add_argument('--num_critic', dest='num_critic', type=int, default=2,
                        help='Discriminator/Generator training Ratio')

    parser.add_argument('--beta_1', dest='beta_1', type=float, default=0.9,
                        help='beta_1 value for adam optimizer')

    parser.add_argument('--beta_2', dest='beta_2', type=float, default=0.999,
                        help='beta_2 value for adam optimizer')

    parser.add_argument('--type_gan', dest='type_gan', choices=['normal', 'wgan', 'wgan-gp'],
                        default='wgan', help='Type of GAN used for training')

    parser.add_argument('--clip_val', dest='clip_val', default=0.01,
                        help='clip discriminator values when using wgan')

    parser.add_argument('--init_method', dest='init_method',
                        choices=['xavier', 'zeros', 'normal'], default='xavier',
                        help='Initialization method for kernel and parameters')

    parser.add_argument('--reg_lambda', dest='reg_lambda', type=float,
                        default=10, help='reg_lambda value used in wgan-gp')

    parser.add_argument('--num_iter', dest='num_iterations', type=int,
                        default=10000, help='Number of iterations to run the GAN')

    # Testing specific

    parser.add_argument('--num_samples', dest='num_samples', type=int,
                        help='Number of samples to generate while testing')

    parser.add_argument('--plot_every_n', type=int, default=500,
                        help='Add plots every n samples')

    args = parser.parse_args()

    num_subpopulations = args.num_subpopulations
    weight_subpopulations = args.weights_subpopulations
    num_markers = args.num_markers
    num_cells = args.num_cells

    # Building the training set

    experiment_name = dt.now().strftime('%d-%m_%H-%M-%S')
    output_dir = os.path.join(args.output_dir, experiment_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build logger
    cellgan_logger = build_logger(
        out_dir=output_dir, logging_format='%(message)s', level=logging.INFO)

    # Generate Gaussian training set
    cellgan_logger.info('Generating gaussian training set...')

    training_data, training_labels = build_gaussian_training_set(
        num_subpopulations=num_subpopulations,
        num_cells=num_cells,
        num_markers=num_markers,
        weights_subpopulations=weight_subpopulations
    )

    start_time = time.time()

    cellgan_logger.info(
        'TIMING: File loading and processing took {} seconds \n'.format(
            datetime.timedelta(seconds=time.time() - start_time)))

    # TODO: Add options for means and variance in the build_gaussian_training_set

    # outlier_scores = compute_outlier_weights(inputs=training_data, method='q_sp')

    # Sampling filters for CellCnn Ensemble
    cellgan_logger.info("Sampling filters for the CellCnn Ensemble...")
    d_filters = get_filters(
        num_cell_cnns=args.num_cell_cnns,
        low=args.d_filters_min,
        high=args.d_filters_max)
    cellgan_logger.info("Filters for the CellCnn Ensemble sampled.\n")

    # Sampling num pooled for CellCnn Ensemble
    cellgan_logger.info("Sampling number of cells to be pooled...")
    d_pooled = get_num_pooled(
        num_cell_cnns=args.num_cell_cnns,
        num_cells_per_input=args.num_cells_per_input)
    cellgan_logger.info("Number of cells to be pooled sampled.\n")

    num_subpopulations = args.num_subpopulations

    if not args.num_experts or args.num_experts <= num_subpopulations:
        num_experts = num_subpopulations
    else:
        num_experts = args.num_experts

    if not args.num_samples:
        num_samples = training_data.shape[0]
    else:
        num_samples = args.num_samples

    # Initialize CellGan
    cellgan_logger.info('Building CellGan...')

    model = CellGan(
        noise_size=args.noise_size,
        moe_sizes=args.moe_sizes,
        batch_size=args.batch_size,
        num_markers=args.num_markers,
        num_experts=num_experts,
        g_filters=args.num_filters_generator,
        d_filters=d_filters,
        d_pooled=d_pooled,
        coeff_l1=args.coeff_l1,
        coeff_l2=args.coeff_l2,
        coeff_act=args.coeff_act,
        num_top=args.num_top,
        dropout_prob=args.dropout_prob,
        noisy_gating=args.noisy_gating,
        noise_eps=args.noise_eps,
        lr=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        reg_lambda=args.reg_lambda,
        clip_val=args.clip_val,
        train=True,
        init_method=args.init_method,
        type_gan=args.type_gan,
        load_balancing=args.load_balancing)

    cellgan_logger.info('CellGan built. \n')

    moe_in_size = model.generator.get_moe_input_size()

    model_hparams = {
        'noise_size': args.noise_size,
        'moe_sizes': [moe_in_size] + args.moe_sizes + [num_markers],
        'batch_size': args.batch_size,
        'num_markers': num_markers,
        'num_experts': num_experts,
        'g_filters': args.num_filters_generator,
        'd_filters_min': args.d_filters_min,
        'd_filters_max': args.d_filters_max,
        'd_filters': d_filters.tolist(),
        'd_pooled': d_pooled.tolist(),
        'coeff_l1': args.coeff_l1,
        'coeff_l2': args.coeff_l2,
        'coeff_act': args.coeff_act,
        'num_top': args.num_top,
        'dropout_prob': args.dropout_prob,
        'noisy_gating': args.noisy_gating,
        'noise_eps': args.noise_eps,
        'lr': args.learning_rate,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'reg_lambda': args.reg_lambda,
        'clip_val': args.clip_val,
        'init_method': args.init_method,
        'type_gan': args.type_gan,
        'load_balancing': args.load_balancing,
        'num_critic': args.num_critic,
        'num_cell_per_input': args.num_cells_per_input,
        'num_cell_cnns': args.num_cell_cnns,

    }

    model_path = os.path.join(output_dir, 'model.ckpt')

    # Write hparams to text file (for reproducibility later)
    cellgan_logger.info('Saving hyperparameters...')
    write_hparams_to_file(out_dir=output_dir, hparams=model_hparams)
    cellgan_logger.info('Hyperparameters saved. \n')

    # Log data to output file
    cellgan_logger.info("Experiment Name: " + experiment_name)
    cellgan_logger.info("Starting our experiments with {} subpopulations".
                        format(num_subpopulations))
    cellgan_logger.info(
        "Number of filters in the CellCnn Ensemble are: {}".format(d_filters))
    cellgan_logger.info(
        "number of cells pooled in the CellCnn Ensemble are: {} \n".format(
            d_pooled))

    # Fit PCA object
    pca = PCA(n_components=2)
    pca = pca.fit(training_data)

    # Fit UMAP object
    um = umap.UMAP()
    um = um.fit(training_data)

    # Training the GAN

    discriminator_loss = list()
    generator_loss = list()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for iteration in range(args.num_iterations):

            if args.subset_sample == 'outlier':
                subset_size = np.random.randint(low=20, high=50)
                outlier_scores = compute_outlier_weights(
                    inputs=training_data, method='q_sp', subset_size=subset_size)

            # Discriminator Training
            model.set_train(True)

            for _ in range(args.num_critic):

                if args.subset_sample == 'outlier':
                    real_batch, indices_batch = \
                        generate_subset(inputs=training_data, num_cells_per_input=args.num_cells_per_input,
                                        weights=outlier_scores, batch_size=args.batch_size,
                                        return_indices=True)

                elif args.subset_sample == 'random':
                    real_batch, indices_batch = \
                        generate_subset(inputs=training_data, num_cells_per_input=args.num_cells_per_input,
                                        weights=None, batch_size=args.batch_size, return_indices=True)

                else:
                    raise NotImplementedError('Subset sampling method {} is not '
                                              'supported'.format(args.subset_sample))

                noise_batch = sample_z(
                    batch_size=args.batch_size,
                    noise_size=args.noise_size,
                    num_cells_per_input=args.num_cells_per_input)

                if args.type_gan == 'wgan':

                    fetches = [model.d_solver, model.d_loss, model.clip_D]
                    feed_dict = {model.Z: noise_batch, model.X: real_batch}

                    _, d_loss, _ = sess.run(
                        fetches=fetches, feed_dict=feed_dict)

                elif args.type_gan == 'normal':

                    fetches = [model.d_solver, model.d_loss]
                    feed_dict = {model.Z: noise_batch, model.X: real_batch}

                    _, d_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

                else:
                    raise NotImplementedError(
                        'Support for wgan-gp not implemented yet.')

            # Generator training
            noise_batch = sample_z(
                batch_size=args.batch_size,
                noise_size=args.noise_size,
                num_cells_per_input=args.num_cells_per_input)

            fetches = [model.g_solver, model.g_loss, model.generator.moe_loss]
            feed_dict = {model.Z: noise_batch, model.X: real_batch}

            _, g_loss, moe_loss = sess.run(
                fetches=fetches, feed_dict=feed_dict)

            discriminator_loss.append(d_loss)
            generator_loss.append(g_loss)

            if iteration % args.plot_every_n == 0:
                model.set_train(False)

                frequency_sampled_batch = compute_frequency(
                    labels=training_labels[indices_batch], weighted=True)

                # Iteration number and losses
                cellgan_logger.info(
                    "We are at iteration: {}".format(iteration + 1))
                if args.subset_sample == 'outlier':
                    cellgan_logger.info("Subset size used: {}".format(subset_size))
                cellgan_logger.info("Discriminator Loss: {}".format(d_loss))
                cellgan_logger.info("Generator Loss: {}".format(g_loss))
                cellgan_logger.info(
                    "Load Balancing Loss: {} \n".format(moe_loss))

                # Actual weights & sampled weights
                cellgan_logger.info("The actual subpopulation weights are: {}".
                                    format(weight_subpopulations))

                if args.subset_sample == 'outlier':
                    cellgan_logger.info("Weights after outlier based sampling: {} \n".format(
                        frequency_sampled_batch))

                else:
                    cellgan_logger.info("Weights after random sampling: {} \n".format(
                        frequency_sampled_batch))

                # Sample fake data for testing
                noise_sample = sample_z(
                    batch_size=1,
                    num_cells_per_input=num_samples,
                    noise_size=args.noise_size)

                fetches = [model.g_sample, model.generator.gates, model.generator.logits]
                feed_dict = {model.Z: noise_sample}

                fake_samples, gates, logits = sess.run(
                    fetches=fetches, feed_dict=feed_dict)
                fake_samples = fake_samples.reshape(num_samples, num_markers)

                fake_sample_experts = np.argmax(gates, axis=1)

                # Sample real data for testing
                real_samples, indices = generate_subset(
                    inputs=training_data,
                    num_cells_per_input=num_samples,
                    weights=None,  # Should I add weights differently?
                    batch_size=1,
                    return_indices=True)
                real_samples = real_samples.reshape(num_samples, num_markers)
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

                cellgan_logger.info("The actual subpopulation weights are: {}".
                                    format(weight_subpopulations))
                cellgan_logger.info(
                    "The learnt subpopulation weights are: {} \n".format(learnt_subpopulation_weights))

                # Save loss plot
                cellgan_logger.info("Saving loss plot")
                plot_loss(
                    out_dir=output_dir,
                    disc_loss=discriminator_loss,
                    gen_loss=generator_loss)
                cellgan_logger.info("Loss plot saved. \n")

                # Plot marker distributions
                cellgan_logger.info("Adding marker distribution plots...")

                plot_marker_distributions(
                    out_dir=output_dir,
                    real_subset=real_samples,
                    fake_subset=fake_samples,
                    real_subset_labels=real_sample_subs,
                    fake_subset_labels=fake_sample_experts,
                    num_experts=num_experts,
                    num_markers=num_markers,
                    num_subpopulations=num_subpopulations,
                    marker_names=DEFAULT_MARKERS,
                    iteration=iteration,
                    logger=cellgan_logger,
                    zero_sub=True)

                cellgan_logger.info("Marker distribution plots added. \n")

                # PCA plot
                cellgan_logger.info("Adding PCA plots...")

                plot_pca(
                    out_dir=output_dir, pca_obj=pca,
                    real_subset=real_samples,
                    fake_subset=fake_samples,
                    real_subset_labels=real_sample_subs,
                    fake_subset_labels=fake_sample_experts,
                    num_experts=num_experts,
                    num_subpopulations=num_subpopulations,
                    iteration=iteration,
                    logger=cellgan_logger,
                    zero_sub=True)

                cellgan_logger.info("PCA plots added. \n")

                # UMAP plot

                cellgan_logger.info("Adding UMAP plots")
                plot_umap(
                    out_dir=output_dir, umap_obj=um,
                    real_subset=real_samples,
                    fake_subset=fake_samples,
                    real_subset_labels=real_sample_subs,
                    fake_subset_labels=fake_sample_experts,
                    num_experts=num_experts,
                    num_subpopulations=num_subpopulations,
                    iteration=iteration,
                    logger=cellgan_logger,
                    zero_sub=True)
                cellgan_logger.info("UMAP plots added. \n")

                # Plotting the heatmap of gating weights
                cellgan_logger.info("Adding Heatmap...")
                plot_heatmap(out_dir=output_dir, logits=logits, fake_subset_labels=fake_sample_experts)
                cellgan_logger.info("Heatmap added")

                # Save the model
                cellgan_logger.info("Saving the model...")
                saver = tf.train.Saver()
                save_path = saver.save(sess, model_path)
                cellgan_logger.info("Model saved at {} \n".format(save_path))

                cellgan_logger.info(
                    "########################################## \n")


if __name__ == '__main__':

    main()
