import numpy as np
import tensorflow as tf
import sys
import os
import json
from argparse import ArgumentParser
from scipy.cluster.hierarchy import linkage, fcluster
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from lib.utils import sample_z, load_model, compute_l2, f_trans, compute_f_measure
from lib.model import CellGan
from lib.preprocessing import read_fcs_data, extract_marker_indices
import matplotlib.pyplot as plt

DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_INPUT_DIR = os.path.join(ROOT_DIR, 'data', DEFAULT_INHIBITOR)
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/bodenmiller/cellgan')
DEFAULT_MARKERS_FILE = os.path.join(DEFAULT_INPUT_DIR, 'markers.csv')


def main():

    parser = ArgumentParser()

    # I/O controls
    parser.add_argument('-e', '--exp_name', default='12-02_13-52-44', help='Experiment Name')

    parser.add_argument('-i', '--inhibitor', default=DEFAULT_INHIBITOR, help='Inhibitor on which training was done')

    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='Where the results are stored')

    parser.add_argument('--markers', dest='marker_file', default=DEFAULT_MARKERS_FILE,
                        help='Filename containing the markers of interest')

    parser.add_argument('--num_samples', help='Number of samples to generate from trained model')

    args = parser.parse_args()

    input_dir = os.path.join(ROOT_DIR, 'data', args.inhibitor)
    results_dir = os.path.join(args.out_dir, args.inhibitor, args.exp_name)

    hparams_file = os.path.join(results_dir, 'Hparams.txt')
    with open(hparams_file, 'r') as f:
        hparams = json.load(f)

    with open(args.marker_file, 'r') as f:
        markers_of_interest = json.load(f)

    all_files = os.listdir(input_dir)
    fcs_files_of_interest = list()

    p = re.compile(hparams['inhibitor_strength'])

    for file in all_files:
        if bool(p.search(file)):
            fcs_files_of_interest.append(file)

    training_data = list()
    training_labels = list()
    # These are not just for training, just for checking later
    celltype_added = 0

    for file in fcs_files_of_interest:

        file_path = os.path.join(input_dir, file.strip())
        fcs_data = read_fcs_data(file_path=file_path)

        try:
            marker_indices = extract_marker_indices(
                fcs_data=fcs_data, markers_of_interest=markers_of_interest)
            num_cells_in_file = fcs_data.data.shape[0]

            if num_cells_in_file >= hparams['subpopulation_limit']:

                processed_data = np.squeeze(fcs_data.data[:, marker_indices])
                processed_data = f_trans(processed_data, c=hparams['cofactor'])

                training_labels.append([celltype_added] * num_cells_in_file)
                celltype_added += 1

                training_data.append(processed_data)

            else:
                continue

        except AttributeError:
            pass

    training_data = np.vstack(training_data)
    training_labels = np.concatenate(training_labels)

    num_subpopulations = len(np.unique(training_labels))

    if not args.num_samples:
        num_samples = len(training_data)

    else:
        num_samples = args.num_samples

    with tf.Session() as sess:

        model = load_model(out_dir=results_dir, session_obj=sess)

        assert isinstance(model, CellGan)

        noise_sample = sample_z(
            batch_size=1,
            num_cells_per_input=num_samples,
            noise_size=hparams['noise_size'])

        fetches = [model.g_sample, model.generator.gates, model.generator.logits]
        feed_dict = {model.Z: noise_sample}

        fake_samples, gates, logits = sess.run(fetches=fetches, feed_dict=feed_dict)
        fake_samples = fake_samples.reshape(num_samples, hparams['num_markers'])
        fake_sample_experts = np.argmax(gates, axis=1)

    centroids = list()
    for expert in np.unique(fake_sample_experts):
        indices = np.flatnonzero(fake_sample_experts == expert)
        centroids.append(np.mean(fake_samples[indices, :], axis=0))

    centroids = np.asarray(centroids)
    Z = linkage(centroids, 'ward')

    for height in range(10):
        cluster_labels = fcluster(Z, height, criterion='distance')
        dists = compute_l2(training_data, centroids)

        indices = np.argmin(dists, axis=1)
        final_clustering = list()
        for index in indices:
            final_clustering.append(cluster_labels[index])

        final_clustering = np.asarray(final_clustering)
        assert len(final_clustering) == len(training_labels)
        print("Height: ", height, " F measure: ", np.round(compute_f_measure(training_labels, final_clustering), 3))


if __name__ == '__main__':
    main()
