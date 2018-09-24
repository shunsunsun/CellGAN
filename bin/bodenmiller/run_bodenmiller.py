import argparse
import numpy as np
import tensorflow as tf
import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import time
import pandas as pd
import logging
from lib.utils import f_trans, get_filters, get_num_pooled, write_hparams_to_file
from lib.utils import generate_subset, sample_z, compute_outlier_weights
from lib.preprocessing import extract_marker_indices, read_fcs_data
from lib.utils import build_logger, save_loss_plot
from lib.model import CellGan
from lib.plotting import plot_marker_distributions
from datetime import datetime as dt
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('-f', '--fcs', dest='fcs_file', default='./data/AKTi/AKTi_fcs.csv',
                        help='file containing names of .fcs files to be used for GAN training')

    parser.add_argument('-m', '--markers', dest='marker_file', default='./data/AKTi/markers.csv',
                        help='Filename containing the markers of interest')

    parser.add_argument('-i', '--in_dir', dest='input_dir', default='./data/AKTi',
                        help='Directory containing the input .fcs files')

    parser.add_argument('-o', '--out_dir', dest='output_dir', default='./results/AKTi',
                        help='Directory where output will be generated.')

    parser.add_argument('-l', '--logging', dest='logging', default=True,
                        help='Whether to log the results to a log file.')

    parser.add_argument('-p', '--plot', dest='to_plot', action='store_true',
                        default=True, help='Whether to plot results')

    # data processing
    parser.add_argument('-a', '--arcsinh', dest='if_arcsinh', action='store_true',
                        help='Whether to use arcsinh transformation on the data')

    parser.add_argument('--sub_limit', dest='subpopulation_limit', type=int,
                        default=30, help='Minimum number of cells to be called a subpopulation')

    parser.add_argument('--cofactor', dest='cofactor', type=int, default=5,
                        help='cofactor for the arcsinh transformation')

    # multi-cell input specific
    parser.add_argument('-b', '--batch_size', dest='batch_size',
                        type=int, default=64, help='batch size used for training')

    parser.add_argument('-nc', '--ncell', dest='num_cells_per_input', type=int,
                        default=100, help='Number of cells per multi-cell input')

    # Generator Parameters
    parser.add_argument('-ns', '--noise_size', dest='noise_size', type=int,
                        default=100, help='Noise dimension for generator')

    parser.add_argument('--moe_sizes', dest='moe_sizes', type=list, default=[100, 100],
                        help='Sizes of the Mixture of Experts hidden layers')

    parser.add_argument('-e', '--experts', dest='num_experts', type=int,
                        help='Number of experts in the generator')

    parser.add_argument('-g', '--g_filters', dest='num_filters_generator',
                        type=int, default=20, help='Number of filters in conv layer of generator')

    parser.add_argument('--n_top', dest='num_top', default=1, type=int,
                        help='Number of experts used for generating each cell')

    parser.add_argument('--noisy_gating', default=True,
                        help='Whether to add noise to gating weights at train time')

    parser.add_argument('--noise_eps', default=1e-2, type=float,
                        help='Noise threshold')

    parser.add_argument('--load_balancing', default=False,
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

    parser.add_argument('-cact', '--coeff_act', dest='coeff_act', default=0,
                        type=float, help='No clue what this is')

    parser.add_argument('-d', '--dropout_prob', dest='dropout_prob',
                        type=float, default=0.5, help='Dropout probability')

    # GAN parameters

    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float,
                        default=2e-4, help='Learning rate for the neural network')

    parser.add_argument('--num_critic', dest='num_critic', type=int,
                        default=2, help='Number of times to train the discriminator before generator')

    parser.add_argument('-b1', '--beta_1', dest='beta_1', type=float,
                        default=0.9, help='beta_1 value for adam optimizer')

    parser.add_argument('-b2', '--beta_2', dest='beta_2', type=float,
                        default=0.999, help='beta_2 value for adam optimizer')

    parser.add_argument('--type_gan', dest='type_gan', choices=['normal', 'wgan', 'wgan-gp'],
                        default='wgan', help='Type of GAN used for training')

    parser.add_argument('--init_method', dest='init_method', choices=['xavier', 'zeros', 'normal'],
                        default='xavier', help='Initialization method for kernel and parameters')

    parser.add_argument('-r', '--reg_lambda', dest='reg_lambda', type=float,
                        default=10, help='reg_lambda value used in wgan-gp')

    parser.add_argument('--num_iter', dest='num_iterations', type=int,
                        default=10000, help='Number of iterations to run the GAN')

    args = parser.parse_args()

    with open(args.fcs_file, 'r') as f:
        fcs_files_of_interest = json.load(f)

    with open(args.marker_file, 'r') as f:
        markers_of_interest = json.load(f)

    # Setup the output directory
    # --------------------------

    experiment_name = dt.now().strftime('%d-%m_%H-%M-%S')
    output_dir = os.path.join(args.output_dir, experiment_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Data Loading Steps
    # ------------------

    print()
    print('Starting to load and process the .fcs files...')
    start_time = time.time()

    training_data = list()
    training_labels = list() # These are not just for training, just for checking later
    celltype_added = 0

    weights_subpopulations = list()

    for file in fcs_files_of_interest:

        file_path = os.path.join(args.input_dir, file.strip())
        fcs_data = read_fcs_data(file_path=file_path)

        try:
            marker_indices = extract_marker_indices(fcs_data=fcs_data, markers_of_interest=markers_of_interest)
            num_cells_in_file = fcs_data.data.shape[0]
            weights_subpopulations.append(num_cells_in_file)

            if args.if_arcsinh and num_cells_in_file >= args.sub_limit:
                processed_data = np.squeeze(fcs_data.data[:, marker_indices])
                processed_data = f_trans(processed_data, c=args.cofactor)

            else:
                processed_data = fcs_data.data[:, marker_indices]

            training_labels.append([celltype_added] * num_cells_in_file)
            celltype_added += 1

            training_data.append(processed_data)

            print('File {} loaded and processed'.format(file))

        except:
            AttributeError

    print('Loading and processing completed.')
    print('Time taken: ', datetime.timedelta(seconds=time.time() - start_time))
    print()

    training_data = np.vstack(training_data)
    training_labels = np.concatenate(training_labels)
    weights_subpopulations = np.array(weights_subpopulations)/np.sum(weights_subpopulations)
    outlier_scores = compute_outlier_weights(inputs=training_data, method='q_sp')

    # Initializing the CellGan model
    # ------------------------------

    d_filters = get_filters(num_cell_cnns=args.num_cell_cnns, low=args.d_filters_min,
                            high=args.d_filters_max)

    d_pooled = get_num_pooled(num_cell_cnns=args.num_cell_cnns,
                              num_cells_per_input=args.num_cells_per_input)

    num_subpopulations = len(np.unique(training_labels))

    if not args.num_experts:
        num_experts = num_subpopulations
    else:
        num_experts = args.num_experts

    print('Building CellGan...')

    model = CellGan(noise_size=args.noise_size, moe_sizes=args.moe_sizes,
                    batch_size=args.batch_size, num_markers=len(markers_of_interest),
                    num_experts=num_experts, g_filters=args.num_filters_generator,
                    d_filters=d_filters, d_pooled=d_pooled, coeff_l1=args.coeff_l1,
                    coeff_l2=args.coeff_l2, coeff_act=args.coeff_act, num_top=args.num_top,
                    dropout_prob=args.dropout_prob, noisy_gating=args.noisy_gating,
                    noise_eps=args.noise_eps, beta_1=args.beta_1, beta_2=args.beta_2,
                    reg_lambda=args.reg_lambda, train=True, init_method=args.init_method,
                    type_gan=args.type_gan, load_balancing=args.load_balancing)

    print('CellGan built. ')
    print()

    moe_in_size = model.generator.get_moe_input_size()

    model_hparams = {
        'noise_size': args.noise_size,
        'num_experts': num_experts,
        'num_top': args.num_top,
        'learning_rate': args.learning_rate,
        'moe_sizes': [moe_in_size] + args.moe_sizes + [len(markers_of_interest)],
        'gfilter': args.num_filters_generator,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'reg_lambda': args.reg_lambda,
        'num_critic': args.num_critic,
        'num_cell_per_input': args.num_cells_per_input,
        'num_cell_cnns': args.num_cell_cnns,
        'type_gan': args.type_gan,
        'd_filters_min': args.d_filters_min,
        'd_filters_max': args.d_filters_max,
        'd_filters': d_filters.tolist(),
        'd_pooled': d_pooled.tolist(),
        'batch_size': args.batch_size,
    }

    model_path = os.path.join(output_dir, 'model.ckpt')

    # Write hparams to text file (for reproducibility later)
    # ------------------------------------------------------

    print('Saving hyperparameters...')
    write_hparams_to_file(out_dir=output_dir, hparams=model_hparams)
    print('Hyperparameters saved.')
    print()

    # Logger properties
    # -----------------

    cellgan_logger = build_logger(out_dir=output_dir, logging_format='%s(message)',
                                  level=logging.INFO)
    
    # Log data to output file
    cellgan_logger.info("Experiment Name: " + experiment_name + '\n')
    cellgan_logger.info("Starting our experiments with {} subpopulations \n".format(num_subpopulations))
    cellgan_logger.info("Number of filters in the CellCnn Ensemble are: {}".format(d_filters))
    cellgan_logger.info("number of Cells Pooled in the CellCnn Ensemble are: {} \n".format(d_pooled))
    cellgan_logger.info("The subpopulation weights are {} \n".format(weights_subpopulations))

    # Training the Gan
    # ----------------

    discriminator_loss = list()
    generator_loss = list()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for iteration in range(args.num_iterations):

            # Discriminator Training
            # ----------------------

            model.set_train(True)

            for _ in range(args.num_critic):

                real_batch = generate_subset(inputs=training_data,
                                             num_cells_per_input=args.num_cells_per_input,
                                             weights=outlier_scores,
                                             batch_size=args.batch_size,
                                             return_indices=False)

                noise_batch = sample_z(batch_size=args.batch_size, noise_size=args.noise_size,
                                       num_cells_per_input=args.num_cells_per_input)

                if args.type_gan == 'wgan':

                    fetches = [model.d_solver, model.d_loss, model.clip_D]
                    feed_dict = {model.Z: noise_batch, model.X: real_batch}

                    _, d_loss, _ = sess.run(fetches=fetches, feed_dict=feed_dict)

                elif args.type_gan == 'normal':

                    fetches = [model.d_solver, model.d_loss]
                    feed_dict = {model.Z: noise_batch, model.X: real_batch}

                    _, d_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

                else:
                    raise NotImplementedError('Support for wgan-gp not implemented yet.')

            # Generator training
            # ------------------

            noise_batch = sample_z(batch_size=args.batch_size, noise_size=args.noise_size,
                                   num_cells_per_input=args.num_cells_per_input)

            fetches = [model.g_solver, model.g_loss, model.generator.moe_loss]
            feed_dict = {model.Z: noise_batch, model.X: real_batch}

            _, g_loss, moe_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

            discriminator_loss.append(d_loss)
            generator_loss.append(g_loss)

            if iteration % 100 == 0:

                model.set_train(False)

                print("We are at iteration: {}".format(iteration + 1))
                print("Discriminator Loss: {}".format(d_loss))
                print("Generator Loss: {}".format(g_loss))
                print("Moe Loss: {}".format(moe_loss))
                print()

                cellgan_logger.info("We are at iteration: {}".format(iteration + 1))
                cellgan_logger.info("Discriminator Loss: {}".format(d_loss))
                cellgan_logger.info("Generator Loss: {}".format(g_loss))
                cellgan_logger.info("Load Balancing Loss: {} \n".format(moe_loss))

                # Fake Data
                # ---------

                num_samples = 1000
                noise_sample = sample_z(batch_size=1, num_cells_per_input=num_samples, noise_size=args.noise_size)

                fetches = [model.g_sample, model.generator.gates]
                feed_dict = {model.Z: noise_sample}

                fake_samples, gates = sess.run(fetches=fetches, feed_dict=feed_dict)
                fake_samples = fake_samples.reshape(num_samples, len(markers_of_interest))

                fake_sample_experts = np.argmax(gates, axis=1)
                num_experts_used = len(np.unique(fake_sample_experts))

                cellgan_logger.info("Number of experts used: {}".format(num_experts_used))

                # Real Data
                # ---------

                real_samples, indices = generate_subset(inputs=training_data,
                                                        num_cells_per_input=num_samples,
                                                        weights=outlier_scores,
                                                        batch_size=1,
                                                        return_indices=True)
                real_samples = real_samples.reshape(num_samples, len(markers_of_interest))
                indices = np.reshape(indices, real_samples.shape[0])
                real_sample_subs = training_labels[indices]

                # Save loss plot
                # --------------

                save_loss_plot(out_dir=output_dir, disc_loss=discriminator_loss,
                               gen_loss=generator_loss)

                # Plot marker distributions
                # -------------------------

                plot_marker_distributions(out_dir=output_dir, real_subset=real_samples,
                                          fake_subset=fake_samples, real_subset_labels=real_sample_subs,
                                          fake_subset_labels=fake_sample_experts, num_experts=num_experts,
                                          num_markers=len(markers_of_interest), num_subpopulations=num_subpopulations,
                                          iteration=iteration, zero_sub=True, pca=False)

                saver = tf.train.Saver()
                save_path = saver.save(sess, model_path)


if __name__ == '__main__':

    main()
