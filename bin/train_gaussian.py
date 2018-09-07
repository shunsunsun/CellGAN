import argparse
import numpy as np
import tensorflow as tf
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from lib.utils import get_filters, get_num_pooled, write_hparams_to_file
from lib.utils import generate_random_subset, sample_z
from lib.utils import build_gaussian_training_set
from lib.model import CellGan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('-o', '--out_dir', dest='output_dir', default='./data/tests/',
                        help='Directory where output will be generated.')

    parser.add_argument('-l', '--logging', dest='logging', default=True,
                        help='Whether to log the results to a log file.')

    parser.add_argument('-p', '--plot', dest='to_plot', action='store_true',
                        default=True, help='Whether to plot results')

    # multi-cell input specific
    parser.add_argument('-b', '--batch_size', dest='batch_size',
                        type=int, default=100, help='batch size used for training')

    parser.add_argument('-nc', '--ncell', dest='num_cells_per_input', type=int,
                        default=100, help='Number of cells per multi-cell input')

    parser.add_argument('-m', '--num_markers', type=int, default=10,
                        help='Number of markers for which we want to generate profiles')

    parser.add_argument('--num_cells', dest='num_cells', type=int,
                        default=40000, help='Total number of cells in the training set')

    parser.add_argument('--num_subpopulations', dest='num_subpopulations', type=int,
                        default=10, help='Number of subpopulations')

    parser.add_argument('-ws', '--weights_subs', dest='weight_subpopulations', type=list,
                        default=[0.1]*10, help='Weights of different subpopulations')

    # Generator Parameters
    parser.add_argument('-ns', '--noise_size', dest='noise_size', type=int,
                        default=100, help='Noise dimension for generator')

    parser.add_argument('--moe_sizes', dest='moe_sizes', type=list, default=[100, 100],
                        help='Sizes of the Mixture of Experts hidden layers')

    parser.add_argument('-e', '--experts', dest='num_experts', type=int,
                        required=True, help='Number of experts in the generator')

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
                        default=0.999, help='beta_1 value for adam optimizer')

    parser.add_argument('-b2', '--beta_2', dest='beta_2', type=float,
                        default=0.9, help='beta_2 value for adam optimizer')

    parser.add_argument('--type_gan', dest='type_gan', choices=['normal', 'wgan', 'wgan-gp'],
                        default='wgan', help='Type of GAN used for training')

    parser.add_argument('--init_method', dest='init_method', choices=['xavier', 'zeros', 'normal'],
                        default='xavier', help='Initialization method for kernel and parameters')

    parser.add_argument('-r', '--reg_lambda', dest='reg_lambda', type=float,
                        default=10, help='reg_lambda value used in wgan-gp')

    parser.add_argument('--num_iter', dest='num_iterations', type=int,
                        default=10000, help='Number of iterations to run the GAN')

    args = parser.parse_args()

    # Building the training set

    training_data, training_labels = build_gaussian_training_set(
        num_subpopulations=args.num_subpopulations,
        num_cells=args.num_cells,
        num_markers=args.num_markers,
        weights_subpopulations=args.weight_subpopulations
    )

    # Getting the number of filters and cells to be pooled for each CellCnn

    d_filters = get_filters(num_cell_cnns=args.num_cell_cnns, low=args.d_filters_min,
                            high=args.d_filters_max)

    d_pooled = get_num_pooled(num_cell_cnns=args.num_cell_cnns,
                              num_cells_per_input=args.num_cells_per_input)

    # Building the CellGan

    print('Building CellGan....')

    model = CellGan(noise_size=args.noise_size, moe_sizes=args.moe_sizes,
                    batch_size=args.batch_size, num_markers=args.num_markers,
                    num_experts=args.num_experts, g_filters=args.num_filters_generator,
                    d_filters=d_filters, d_pooled=d_pooled, coeff_l1=args.coeff_l1,
                    coeff_l2=args.coeff_l2, coeff_act=args.coeff_act, num_top=args.num_top,
                    dropout_prob=args.dropout_prob, noisy_gating=args.noisy_gating,
                    noise_eps=args.noise_eps, beta_1=args.beta_1, beta_2=args.beta_2,
                    reg_lambda=args.reg_lambda, train=True, init_method=args.init_method,
                    type_gan=args.type_gan, load_balancing=args.load_balancing)

    print('CellGan built. ')
    print()

    moe_in_size = model.generator.get_moe_input_size()

    # Writing hyperparameters to dictionary for model loading later

    model_hparams = {
        'noise_size': args.noise_size,
        'num_experts': args.num_experts,
        'num_top': args.num_top,
        'learning_rate': args.learning_rate,
        'moe_sizes': [moe_in_size] + args.moe_sizes + [args.num_markers],
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    write_hparams_to_file(out_dir=args.output_dir, hparams=model_hparams)

    # Training the GAN

    discriminator_loss = list()
    generator_loss = list()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for iteration in range(args.num_iterations):

            # ----------------------
            # DISCRIMINATOR TRAINING
            # ----------------------

            model.set_train(True)

            for _ in range(args.num_critic):

                train_real, _ = generate_random_subset(inputs=training_data,
                                                       num_cells_per_input=args.num_cells_per_input,
                                                       batch_size=args.batch_size)

                # TODO: Add better name instead of train_real

                training_noise = sample_z(batch_size=args.batch_size, noise_size=args.noise_size,
                                          num_cells_per_input=args.num_cells_per_input)

                if args.type_gan == 'wgan':

                    fetches = [model.d_solver, model.d_loss, model.clip_D]
                    feed_dict = {model.Z: training_noise, model.X: train_real}

                    _, d_loss, _ = sess.run(fetches=fetches, feed_dict=feed_dict)

                elif args.type_gan == 'normal':

                    fetches = [model.d_solver, model.d_loss]
                    feed_dict = {model.Z: training_noise, model.X: train_real}

                    _, d_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

                else:
                    raise NotImplementedError('Support for wgan-gp not implemented yet.')

            # ------------------
            # GENERATOR TRAINING
            # ------------------

            training_noise = sample_z(batch_size=args.batch_size, noise_size=args.noise_size,
                                      num_cells_per_input=args.num_cells_per_input)

            fetches = [model.g_solver, model.g_loss, model.generator.moe_loss]
            feed_dict = {model.Z: training_noise, model.X: train_real}

            _, g_loss, moe_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

            discriminator_loss.append(d_loss)
            generator_loss.append(g_loss)

            # TODO: Add the plotting thing

            # if it % 100 == 0:
            #     model.set_train(False)
            #
            #     print("We are at iteration: {}".format(it + 1))
            #     # print("Discriminator Loss: {}".format(d_loss))
            #     # print("Generator Loss: {}".format(g_loss))
            #     # print("Moe Loss: {}".format(moe_loss))
            #
            #     logger.info("We are at iteration: {}".format(it + 1))
            #     logger.info("Discriminator Loss: {}".format(d_loss))
            #     logger.info("Generator Loss: {}".format(g_loss))
            #     logger.info("Load Balancing Loss: {} \n".format(moe_loss))
            #
            #     # Fake Data
            #     # ---------
            #
            #     n_samples = 1000
            #     input_noise = sample_z(shape=[1, n_samples, noise_size])
            #
            #     fetches = [model.g_sample, model.generator.gates]
            #     feed_dict = {
            #         model.Z: input_noise
            #     }
            #
            #     test_fake, gates = sess.run(
            #         fetches=fetches,
            #         feed_dict=feed_dict
            #     )
            #
            #     test_fake = test_fake.reshape(n_samples, n_markers)
            #
            #     experts = np.argmax(gates, axis=1)
            #     experts_used = len(np.unique(experts))
            #
            #     # logger.info("Number of experts used: {}".format(experts_used))
            #
            #     # Real Data
            #     # ---------
            #
            #     test_real, indices = generate_random_subset(inputs=real_data,
            #                                                 ncell=n_samples,
            #                                                 batch_size=1)
            #
            #     test_real = test_real.reshape(n_samples, n_markers)
            #     indices = np.reshape(indices, test_real.shape[0])
            #     test_real_subs = real_subs[indices]
            #
            #     # This is just for understanding, not used anywhere in training the network
            #
            #     logger.info("Number of subpopulations present: {} \n".format(len(np.unique(test_real_subs))))
            #
            #     test_data_freq = dict(Counter(test_real_subs))
            #     total = sum(test_data_freq.values(), 0.0)
            #     test_data_freq = {k: round(test_data_freq[k] / total, 3) for k in sorted(test_data_freq.keys())}
            #
            #     real_data_freq = dict(Counter(real_subs))
            #     total = sum(real_data_freq.values(), 0.0)
            #     real_data_freq = {k: round(real_data_freq[k] / total, 3) for k in sorted(real_data_freq.keys())}
            #
            #     fake_data_freq = dict(Counter(experts))
            #     total = sum(fake_data_freq.values(), 0.0)
            #     fake_data_freq = {k: round(fake_data_freq[k] / total, 3) for k in sorted(fake_data_freq.keys())}
            #
            #     logger.info("Frequency of subpopulations in real data is: {}".format(real_data_freq))
            #     logger.info("Frequency of subpopulations in test data is: {} \n".format(test_data_freq))
            #
            #     logger.info("Checking Weights")
            #     logger.info("Real subpopulation weights: {}".format(sorted(real_data_freq.values(), reverse=True)))
            #     logger.info("Expert subpopulation weights: {}\n".format(sorted(fake_data_freq.values(), reverse=True)))
            #
            #     logger.info("-----------\n")
            #
            #     pca_plot(
            #         out_dir=output_dir,
            #         real_data=test_real,
            #         fake_data=test_fake,
            #         experts=experts,
            #         real_subs=test_real_subs,
            #         it=it
            #     )
            #
            #     saveLossPlot(
            #         dir_output=output_dir,
            #         disc_loss=D_loss,
            #         gen_loss=G_loss
            #     )
            #
            #     saver = tf.train.Saver()
            #     save_path = saver.save(sess, model_path)


if __name__ == '__main__':

    main()