""" Hierarchical clustering of the generated data from experts """

import numpy as np
import tensorflow as tf
import argparse
import os
import json
from scipy.cluster.hierarchy import fcluster, linkage

from cellgan.lib.utils import load_model, sample_z, compute_l2, compute_l1, compute_f_measure, compute_f_measure_uniformly_weighted
from cellgan.lib.data_utils import get_fcs_filenames, load_fcs

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", dest="input_dir", default="./data/", help="Data directory")
    parser.add_argument("--log_dir", dest="results_dir", default="./results/cellgan", help="Results directory")
    parser.add_argument("--inhibitor", default="AKTi", help="Inhibitor used.")
    parser.add_argument("--exp_name", default="15_02_2019-09_28_05", help="Name of experiment to compute clustering for.")
    parser.add_argument("--cofactor", default=5)
    parser.add_argument("--sub_limit", dest="subpopulation_limit", default=30)
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, args.inhibitor, args.exp_name)
    hparams_file = os.path.join(out_dir, "Hparams.txt")
    markers_file = os.path.join(out_dir, "markers.csv")
    fcs_file = os.path.join(out_dir, "fcs.csv")

    def load(filename):
        with open(filename, "r") as f:
            vals = json.load(f)
        return vals

    hparams = load(hparams_file)
    markers_of_interest = load(markers_file)
    fcs_files_of_interest = load(fcs_file)

    num_experts = hparams["num_experts"]
    max_iteration_saved = max([int(number) for number in os.listdir(out_dir) if number[0].isdigit()])
    training_data, training_labels = load_fcs(fcs_files_of_interest, markers_of_interest, args)
    num_samples = training_data.shape[0]

    with tf.Session() as sess:
        model = load_model(out_dir=out_dir, session_obj=sess, iteration=max_iteration_saved)

        noise_sample = sample_z(
            batch_size=1,
            num_cells_per_input=num_samples,
            noise_size=hparams["noise_size"])

        fetches = [model.g_sample, model.generator.gates, model.generator.logits]
        feed_dict = {model.Z: noise_sample}
        fake_samples, gates, logits = sess.run(fetches=fetches, feed_dict=feed_dict)
        fake_samples = fake_samples.reshape(num_samples, hparams["num_markers"])
        fake_sample_experts = np.argmax(gates, axis=1)

        means = list()
        for expert in range(num_experts):
            indices = np.flatnonzero(fake_sample_experts == expert)
            means.append(np.mean(fake_samples[indices], axis=0))
        means = np.array(means)

        Z = linkage(means, 'ward') # Ward linkage for clustering means
        dists = compute_l2(training_data, means)
        closest_experts = np.argmin(dists, axis=1)

        f_scores = list()
        for height in range(num_experts):
            clusters = fcluster(Z, height, criterion='distance')

            cluster_labels = list()
            for i in range(num_samples):
                cluster_labels.append(clusters[closest_experts[i]])
            f_scores.append(compute_f_measure_uniformly_weighted(training_labels, cluster_labels))

        print(np.max(f_scores))

if __name__ == "__main__":
    main()
