import argparse
import numpy as np
import tensorflow as tf
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import time
import datetime
import pandas as pd
from FlowCal.io import FCSData
from lib.utils import f_trans, get_filters, get_num_pooled
from lib.model import CellGan


def main():

    parser = argparse.ArgumentParser()

    # loading and saving data
    parser.add_argument('-f', '--fcs', dest='fcs_file', default='./data/fcs.csv',
                        help='file containing names of .fcs files to be used for GAN training')

    parser.add_argument('-m', '--markers', dest='marker_file', default='./data/markers.csv',
                        help='Filename containing the markers of interest')

    parser.add_argument('-i', '--in_dir', dest='input_dir', default='./data/AKTi',
                        help='Directory containing the input .fcs files')

    parser.add_argument('-o', '--out_dir', dest='output_dir', default='./data/tests/NK_test',
                        help='Directory where output will be generated.')

    parser.add_argument('-p', '--plot', dest='to_plot', action='store_true',
                        default=True, help='Whether to plot results')

    # data processing
    parser.add_argument('-a', '--arcsinh', dest='if_arcsinh', action='store_true',
                        help='Whether to use arcsinh transformation on the data')

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

    args = parser.parse_args()

    # Choose a value for n_critic

    if args.type_gan == 'wgan-gp':
        n_critic = 3

    elif args.type_gan == 'wgan':
        n_critic = 2

    elif args.type_gan == 'normal':
        n_critic = 1

    fcs_files_of_interest = list(pd.read_csv(args.fcs_file, sep=','))
    markers_of_interest = list(pd.read_csv(args.marker_file, sep=','))

    # Add Data Loading Steps next here

    print()
    print('Starting to load and process the .fcs files...')
    start_time = time.time()

    loaded_files = dict()

    for file in fcs_files_of_interest:
        file_path = os.path.join(args.input_dir, file.strip())
        data = FCSData(file_path)

        if args.if_arcsinh:

            data = f_trans(data, c=args.cofactor)

        loaded_files[file] = data

        print('File {} loaded and processed'.format(file))

    print('Loading and processing completed.')
    print('Time taken: ', datetime.timedelta(seconds=time.time() - start_time))
    print()

    # TODO: Need to add the data loading pipeline properly
    # TODO: Add a command line interface for Gaussian models as well

    # Initializing our CellGan model
    # ------------------------------

    d_filters = get_filters(num_cell_cnns=args.num_cell_cnns, low=args.d_filters_min,
                            high=args.d_filters_max)

    d_pooled = get_num_pooled(num_cell_cnns=args.num_cell_cnns,
                              num_cells_per_input=args.num_cells_per_input)

    print('Building CellGan...')
    print('-------------------')

    model = CellGan(noise_size=args.noise_size, moe_sizes=args.moe_sizes,
                    batch_size=args.batch_size, num_markers=len(markers_of_interest),
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


if __name__ == '__main__':

    main()
