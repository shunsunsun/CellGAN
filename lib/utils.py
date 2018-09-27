
import tensorflow as tf
import numpy as np
import os
import json

import matplotlib
matplotlib.use('Agg')
from collections import Counter
import logging
from scipy.stats import ks_2samp

xav_init = tf.contrib.layers.xavier_initializer
normal_init = tf.truncated_normal_initializer
zero_init = tf.zeros_initializer

# Different methods for initializing the data
initializers = dict()
initializers['xavier'] = xav_init
initializers['normal'] = normal_init
initializers['zeros'] = zero_init

DEFAULT_SUBSET_SIZE = 20


def f_trans(x, c):
    """
    Computes a transformation of the input flow cytometry data for downstream analysis
    :param x: np.ndarray, data from flow cytometry experiment
    :param c: float, cofactor
    :return: transformed data
    """
    return np.arcsinh(1./c * x)


def get_filters(num_cell_cnns, low, high):

    """
    Get a list of number of filters to be used for the CellCnn Ensemble.
    The filters are chosen randomly between limits specified by low and high

    :param num_cell_cnns: int, number of CellCnns in the ensemble
    :param low: int, minimum permissible number of filters for a CellCnn
    :param high: int, maximum permissible number of filters for a CellCnn
    :return: filters, a numpy array of size num_cell_cnns
    """

    filters = np.random.randint(low=low, high=high, size=num_cell_cnns)

    return filters


def get_num_pooled(num_cell_cnns, num_cells_per_input):

    """
    Returns a list of number of cells to be pooled

    :param num_cell_cnns: int, number of CellCnns in the ensemble
    :param num_cells_per_input: int, number of cells per multi-cell input
    :return: num_pooled, a numpy array of size num_cell_cnns
    """

    num_pooled = np.random.randint(low=1, high=num_cells_per_input, size=num_cell_cnns)

    return num_pooled


def sample_z(batch_size, num_cells_per_input, noise_size):
    
    """
    Generates noise, which is input to the generator based on given shape
    :param batch_size: int, mini-batch size
    :param num_cells_per_input: int, number of cells per multi-cell input
    :param noise_size: int, noise dimension 
    :return: noise, a numpy array of shape (batch_size, num_cells, noise_size)
    """

    noise = np.random.multivariate_normal(
        mean=np.zeros(shape=noise_size),
        cov=np.eye(N=noise_size),
        size=(batch_size, num_cells_per_input)
    )

    return noise


def build_gaussian_training_set(num_subpopulations, num_cells, weights_subpopulations, num_markers, shuffle=False):
    
    """
    Build a training set of desired number of subpopulations and number of cells in the training set.
    The number of markers and the weights of the subpopulations are also specified. Returns each cell's
    marker profile and the subpopulation it belongs to 
    
    :param num_subpopulations: int, number of subpopulations in the data
    :param num_cells: int, total number of cells in the 
    :param weights_subpopulations: list of floats, weights of different subpopulations
    :param num_markers: int, number of markers per cell measured
    :param shuffle: bool, default false, whether to shuffle the training set
    :return: data, y_sub_populations
    """

    means = list()
    sd = list()

    for i in range(num_subpopulations):
        means.append(np.random.choice(range(1, 5), num_markers, replace=True))
        sd.append(np.random.sample(num_markers))

    data = list()
    y_sub_populations = list()
    
    for i in range(num_subpopulations):
        temp_num_cells = int(num_cells * weights_subpopulations[i])
        data.append(np.random.normal(means[i], sd[i], (temp_num_cells, num_markers)))
        y_sub_populations.append([i + 1] * temp_num_cells)
    data = np.vstack(data)
    y_sub_populations = np.concatenate(y_sub_populations)
    
    if shuffle:
        ind_shuffle = np.random.choice(range(num_cells), num_cells, replace=False)
        data = data[ind_shuffle, :]
        y_sub_populations = y_sub_populations[ind_shuffle]

    return data, y_sub_populations


def get_batches(inputs, batch_size, num_batches, num_cells_per_input, weights=None):

    """
    Generate multiple batches of training data from given inputs
    :param inputs: input numpy array of shape (num_cells, num_markers)
    :param batch_size: int, batch_size of each batch generated
    :param num_batches: int, number of batches of training data to generate
    :param num_cells_per_input: int, number of cells per multi-cell input
    :param weights: list of float, whether there is a preference for some cells
    :return:
    """

    batches = [generate_subset(inputs=inputs, num_cells_per_input=num_cells_per_input,
                               batch_size=batch_size, weights=weights, return_indices=False)
               for _ in range(num_batches)]
    return batches


def generate_subset(inputs, num_cells_per_input, batch_size, weights=None, return_indices=False):

    """
    Returns a random subset from input data of shape (batch_size, num_cells_per_input, num_markers)
    :param inputs: numpy array, the input ndarray to sample from
    :param num_cells_per_input: int, number of cells per multi-cell input
    :param batch_size: int, batch size of the subset
    :param weights: list of float, whether there is a preference for some cells
    :param return_indices: bool, whether to return subset indices or not
    :return:
    """

    num_cells_total = inputs.shape[0]

    if weights is not None:
        indices = np.random.choice(num_cells_total, size=batch_size * num_cells_per_input, replace=True, p=weights)

    else:
        indices = np.random.choice(num_cells_total, size=batch_size * num_cells_per_input, replace=True)

    subset = inputs[indices, ]
    subset = np.reshape(subset, newshape=(batch_size, num_cells_per_input, -1))

    if return_indices:
        return subset, indices

    else:
        return subset


def compute_outlier_weights(inputs, method='q_sp'):

    """
    Returns the outlier weights computed for the inputs using the method specified
    :param inputs: np.ndarray, dataset comprised of cells to be used for training
    :param method: what method to use for outlier computation (default q_sp)
    :return:
    """

    if method != 'q_sp':
        raise NotImplementedError('Other outlier methods are not implemented currently')

    else:
        subset_size = DEFAULT_SUBSET_SIZE

        if subset_size < inputs.shape[0]:
            subset_indices = np.random.choice(inputs.shape[0], size=subset_size, replace=False)
        else:
            subset_indices = np.random.choice(inputs.shape[0], size=subset_size, replace=True)

        sampled_subset = inputs[subset_indices, :]
        dists = np.zeros(inputs.shape[0])

        for index in range(len(dists)):
            dists[index] = np.min(np.square(inputs[index] - sampled_subset))

        outlier_weights = dists/dists.sum()

        return outlier_weights


def write_hparams_to_file(out_dir, hparams):

    """
    Writes hyperparameters used in experiment to specified output directory

    :param out_dir: str, directory name
    :param hparams: dictionary, keys as hyperparameter names
    :return: no returns
    """

    filename = os.path.join(out_dir, 'Hparams.txt')

    with open(filename, 'w') as f:

        f.write(json.dumps(hparams))


def build_logger(out_dir, level=logging.INFO, logging_format='%(message)s'):

    """
    Setup the logger
    :param out_dir: Output directory
    :param level: Logger level (One of INFO, DEBUG)
    :param logging_format: What format to use for logging messages
    :return: logger with properties defined above
    """

    log_file_name = os.path.join(out_dir, 'Output.log')
    logger = logging.getLogger('CellGan')

    handler = logging.FileHandler(log_file_name, mode='w')
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger


def compute_frequency(labels, weighted=False):

    labels = labels.reshape(-1)

    label_counts = Counter(labels)

    if not weighted:
        return label_counts

    else:
        label_sum = np.sum(list(label_counts.values()))
        for key in label_counts:
            label_counts[key] = label_counts[key]/label_sum

        return label_counts


def compute_ks(real_data, real_labels, fake_data, expert_labels, num_subpopulations, num_experts):

    ks_sums = list()

    for expert in range(num_experts):

        ks_sum_per_expert = list()
        expert_indices = np.where(expert_labels == expert)[0]

        if len(expert_indices) == 0:
            ks_sums.append([0]*num_subpopulations)
            continue

        else:
            fake_data_by_expert = fake_data[expert_indices, :]

            for sub in range(num_subpopulations):
                subs_indices = np.where(real_labels == sub)[0]

                if len(subs_indices) == 0:
                    ks_sum_per_expert.append(0)
                    continue

                else:
                    real_data_by_sub = real_data[subs_indices]
                    ks_sum = np.sum([ks_2samp(real_data_by_sub[:, marker], fake_data_by_expert[:, marker])[0]
                                     for marker in range(real_data.shape[-1])])

                    ks_sum_per_expert.append(ks_sum)

        ks_sums.append(ks_sum_per_expert)

    ks_sums = np.asarray(ks_sums)
    assert ks_sums.shape == (num_experts, num_subpopulations)

    return ks_sums


def assign_expert_to_subpopulation(real_data, real_labels, fake_data,
                                   expert_labels, num_subpopulations, num_experts):

    ks_sums = compute_ks(real_data=real_data, real_labels=real_labels, fake_data=fake_data,
                         expert_labels=expert_labels, num_subpopulations=num_subpopulations,
                         num_experts=num_experts)

    expert_assignments = np.argmax(ks_sums, axis=1)

    return expert_assignments


def compute_learnt_subpopulation_weights(expert_labels, expert_assignments, num_subpopulations):

    expert_weights = compute_frequency(labels=expert_labels, weighted=True)
    learnt_subpopulation_weights = dict()

    for subpopulation in range(num_subpopulations):

        if subpopulation not in expert_assignments:
            learnt_subpopulation_weights[subpopulation] = 0
        else:
            which_experts = np.where(expert_assignments == subpopulation)[0]
            for expert in which_experts:
                learnt_subpopulation_weights += expert_weights[expert]

    return learnt_subpopulation_weights
