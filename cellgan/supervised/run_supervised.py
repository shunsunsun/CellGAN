import numpy as np
import tensorflow as tf
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime

from cellgan.supervised.defaults import *
from cellgan.lib.data_utils import load_fcs
from cellgan.supervised.model import Trainer
from cellgan.lib.utils import compute_f_measure_uniformly_weighted
from scipy.cluster.hierarchy import fcluster, linkage

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--inhibitor", default="AKTi", help="Inhibitor used")
    parser.add_argument("--exp_name", default="22_02_2019-09_32_42", help="Experiment name")
    parser.add_argument("--in_dir", dest="input_dir", default=DATA_DIR, help="Directory of files")
    parser.add_argument("--cofactor", default=5, type=int, help="Cofactor")
    parser.add_argument("--subpopulation_limit", default=30, type=int, help="SUb limit")
    parser.add_argument("--num_iter", default=1000, type=int)
    parser.add_argument("--print_every_n", default=200, type=int)
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)

    args = parser.parse_args()

    # out_dir is for the trained GAN model, save_dir is to save results from supervised learning
    out_dir = os.path.join(RESULTS_DIR, args.inhibitor, args.exp_name)
    fcs_savefile = os.path.join(out_dir, 'fcs.csv')
    markers_savefile = os.path.join(out_dir, 'markers.csv')
    hparams_file = os.path.join(out_dir, 'Hparams.txt')
    experiment_name = dt.now().strftime("%d_%m_%Y-%H_%M_%S")

    save_dir = os.path.join(RESULTS_DIR, "supervised", experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(hparams_file, 'r') as f:
        hparams = json.load(f)

    with open(fcs_savefile, 'r') as f:
        fcs_files_of_interest = json.load(f)

    with open(markers_savefile, 'r') as f:
        markers_of_interest = json.load(f)

    test_data, test_labels = load_fcs(fcs_files_of_interest, markers_of_interest, args, logger=None)

    with tf.Session() as sess:

        f_measures = list()

        trainer = Trainer(exp_name=args.exp_name, iteration=6001, sess_obj=sess, inhibitor=args.inhibitor, lr=args.learning_rate)
        sess.run(tf.global_variables_initializer())
        losses, mean_fs, std_fs = trainer.fit(X=test_data, y=test_labels, num_iterations=args.num_iter, print_every_n=args.print_every_n)
        preds = trainer.predict(test_data)

        fake_samples, fake_sample_experts = trainer._generate_samples(num_samples=10000)
        means = list()
        for expert in range(hparams['num_experts']):
            indices = np.flatnonzero(fake_sample_experts == expert)
            means.append(np.mean(fake_samples[indices], axis=0))
        means = np.array(means)

        Z = linkage(means, 'ward')

        f_scores = list()
        for height in range(hparams['num_experts']):
            clusters = fcluster(Z, height, criterion='distance')

            cluster_labels = list()
            for i in range(len(test_data)):
                cluster_labels.append(clusters[preds[i]])
            f_scores.append(compute_f_measure_uniformly_weighted(test_labels, cluster_labels))

    f_measures.append(max(f_scores))

    print("Mean: ", np.mean(f_measures))
    print("Std: ", np.std(f_measures))

    iterations = np.range(0, args.num_iter, step=args.print_every_n)
    plt.figure()
    plt.plot(iterations, losses)
    plt.xlabel("Iteration Number")
    plt.ylabel("Loss value")
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(iterations, mean_fs)
    plt.xlabel("Iteration Number")
    plt.ylabel("Averga F-measure")
    plt.savefig(os.path.join(save_dir, "F-measure.jpg"))
    plt.close()


if __name__ == '__main__':
    main()