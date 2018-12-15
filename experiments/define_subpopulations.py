import numpy as np
import tensorflow as tf
import sys
import os
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from argparse import ArgumentParser
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)


from lib.preprocessing import extract_marker_indices, read_fcs_data
from lib.utils import sample_z, compute_wasserstein, load_model, compute_ks, build_logger, f_trans
from lib.utils import generate_subset
from lib.plotting import plot_marker_distributions
from lib.model import CellGan
import matplotlib.pyplot as plt

DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_INPUT_DIR = os.path.join(ROOT_DIR, 'data', DEFAULT_INHIBITOR)
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/bodenmiller/cellgan')
DEFAULT_MARKERS_FILE = os.path.join(DEFAULT_INPUT_DIR, 'markers.csv')

def compute_variance(data, cluster_labels):
    """Computes variance in the clustering """

    unique_labels = np.unique(cluster_labels)
    variances = list()

    for label in unique_labels:
        indices = np.flatnonzero(cluster_labels == label)
        data_for_cluster = data[indices, :]
        variances.append(data_for_cluster.var())

    return np.min(variances)

def plot_cluster_variance(variances):
    
    plt.plot(np.arange(1, len(variances) + 1), variances)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Variance of computed clusters')
    plt.savefig('Cluster_variance.jpg')

def main():

    parser = ArgumentParser()

    # I/O controls
    parser.add_argument('-e', '--exp_name', default='17-11_19-09-11', help='Experiment Name')

    parser.add_argument('-i', '--inhibitor', default=DEFAULT_INHIBITOR, help='Inhibitor on which training was done')

    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='Where the results are stored')

    parser.add_argument('--markers', dest='marker_file', default=DEFAULT_MARKERS_FILE, 
        help='Filename containing the markers of interest')

    parser.add_argument('--num_samples', help='Number of samples to generate from trained model')

    args = parser.parse_args()

    input_dir = os.path.join(ROOT_DIR, 'data', args.inhibitor)
    results_dir = os.path.join(args.out_dir, args.inhibitor, args.exp_name)

    # Load Hyperparameters
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

        # Sample real data for testing
        real_samples, indices = generate_subset(
            inputs=training_data,
            num_cells_per_input=num_samples,
            weights=None,  #Should I add weights differently?
            batch_size=1,
            return_indices=True)
        real_samples = real_samples.reshape(num_samples,len(markers_of_interest))
        indices = np.reshape(indices, real_samples.shape[0])
        real_sample_subs = training_labels[indices]
    
    variances = list()

    for k in range(1, hparams['num_experts'] + 1):

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(fake_samples)

        cluster_labels = kmeans.predict(fake_samples)
        var = compute_variance(fake_samples, cluster_labels)
        variances.append(var)

    # plot_cluster_variances(variances)
    print(", k_min: ", 1 + np.argmin(variances))
    k_min = 1 + np.argmin(variances)

    cluster_labels  = AgglomerativeClustering(n_clusters=k_min).fit_predict(fake_samples)
    clustering_dir = os.path.join(results_dir, 'clustering')

    plot_marker_distributions(
        out_dir=clustering_dir,
        real_subset=real_samples,
        fake_subset=fake_samples,
        real_subset_labels=real_sample_subs,
        fake_subset_labels=cluster_labels,
        num_experts=len(np.unique(cluster_labels)),
        num_markers=len(markers_of_interest),
        num_subpopulations=num_subpopulations,
        marker_names=markers_of_interest,
        iteration=0,
        logger=None,
        zero_sub=True)


if __name__ == '__main__':
    main()
