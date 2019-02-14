import tensorflow as tf
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import logging
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score

from cellgan.lib.model import CellGan

xav_init = tf.contrib.layers.xavier_initializer
normal_init = tf.truncated_normal_initializer
zero_init = tf.zeros_initializer

DEFAULT_SUBSET_SIZE = 50

# Different methods for initializing the data
initializers = dict()
initializers['xavier'] = xav_init
initializers['normal'] = normal_init
initializers['zeros'] = zero_init


def f_trans(x, c):
    """
    Computes a transformation of the input flow cytometry data for downstream analysis
    :param x: np.ndarray, data from flow cytometry experiment
    :param c: float, cofactor
    :return: transformed data
    """
    return np.arcsinh(x / c)


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

    num_pooled = np.random.randint(
        low=1, high=num_cells_per_input, size=num_cell_cnns)

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
        size=(batch_size, num_cells_per_input))

    return noise


def build_gaussian_training_set(num_subpopulations,
                                num_cells,
                                weights_subpopulations,
                                num_markers,
                                shuffle=False):
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
        means.append(np.random.choice(range(1, 10), num_markers, replace=True))
        sd.append(np.random.sample(num_markers))

    data = list()
    y_sub_populations = list()

    for i in range(num_subpopulations):
        temp_num_cells = int(num_cells * weights_subpopulations[i])
        data.append(
            np.random.normal(means[i], sd[i], (temp_num_cells, num_markers)))
        y_sub_populations.append([i] * temp_num_cells)
    data = np.vstack(data)
    y_sub_populations = np.concatenate(y_sub_populations)

    if shuffle:
        ind_shuffle = np.random.choice(
            range(num_cells), num_cells, replace=False)
        data = data[ind_shuffle, :]
        y_sub_populations = y_sub_populations[ind_shuffle]

    return data, y_sub_populations


def get_batches(inputs,
                batch_size,
                num_batches,
                num_cells_per_input,
                weights=None):
    """
    Generate multiple batches of training data from given inputs
    :param inputs: input numpy array of shape (num_cells, num_markers)
    :param batch_size: int, batch_size of each batch generated
    :param num_batches: int, number of batches of training data to generate
    :param num_cells_per_input: int, number of cells per multi-cell input
    :param weights: list of float, whether there is a preference for some cells
    :return:
    """

    batches = [
        generate_subset(
            inputs=inputs,
            num_cells_per_input=num_cells_per_input,
            batch_size=batch_size,
            weights=weights,
            return_indices=False) for _ in range(num_batches)
    ]
    return batches


def generate_subset(inputs,
                    num_cells_per_input,
                    batch_size,
                    weights=None,
                    return_indices=False):
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
        indices = np.random.choice(
            num_cells_total,
            size=batch_size * num_cells_per_input,
            replace=True,
            p=weights)

    else:
        indices = np.random.choice(
            num_cells_total,
            size=batch_size * num_cells_per_input,
            replace=True)

    subset = inputs[indices, ]
    subset = np.reshape(subset, newshape=(batch_size, num_cells_per_input, -1))

    if return_indices:
        return subset, indices

    else:
        return subset


def compute_outlier_weights(inputs,
                            method='q_sp',
                            metric='l2',
                            subset_size=DEFAULT_SUBSET_SIZE):
    """
    Computes the outlier weights using the given metric
    :param inputs: np.ndarray, dataset comprised of cells to be used for training
    :param method: what method to use for outlier computation (default q_sp)
    :param subset_size: Size of the randomly sampled subset to compute outliers
    :return: outlier_weights
    """

    if method != 'q_sp':
        raise NotImplementedError(
            'Other outlier methods are not implemented currently')

    else:
        outlier_weights = q_sp(inputs, metric=metric, subset_size=subset_size)

        return outlier_weights


def q_sp(inputs, subset_size=DEFAULT_SUBSET_SIZE, metric='l2'):
    """q_sp method for computing outlier weights
    :param inputs: Input dataset
    :param subset_size: Size of the randomly sampled subset to compute outliers
    :param metric: Which distance metric to use (default l2)
    :return outlier_weights 
    """

    if subset_size < inputs.shape[0]:
        subset_indices = np.random.choice(
            inputs.shape[0], size=subset_size, replace=False)
    else:
        subset_indices = np.random.choice(
            inputs.shape[0], size=subset_size, replace=True)

    sampled_subset = inputs[subset_indices, :]
    dists = compute_closest(inputs, sampled_subset, metric=metric)
    dists[dists <= 0] = 0
    outlier_weights = dists / dists.sum()

    return outlier_weights


def compute_l2(x, y):
    """Vectorized implementation of l2-norm"""

    dists = np.zeros((x.shape[0], y.shape[0]))

    x_squared = np.sum(np.square(x), axis=1)
    y_squared = np.sum(np.square(y), axis=1)

    dists += x_squared[:, np.newaxis]
    dists += y_squared[np.newaxis, :]
    x_dot_y = np.dot(x, y.T)

    assert dists.shape == x_dot_y.shape

    dists -= 2 * x_dot_y

    return dists


def compute_l1(x, y):
    """Faster implementation of l1-norm"""

    dists = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        sample = x[i, :]
        sample_dists = np.abs(sample - y)
        sample_dists = np.sum(sample_dists, axis=1)

        dists[i] = sample_dists

    return dists


def compute_closest(x, y, metric='l2'):
    """Finds smallest distance to y for x (x is many data points)
    :param x: data point for which we want to compute the closest distance to
    :param y: subset in which we wish to find smallest distance to (Matrix)
    :param metric: Which distance metric to use (default l2)
    :return: smallest_dist
    """

    if metric == 'l2':

        dists = compute_l2(x, y)
        smallest_dist = np.min(dists, axis=1)
        assert len(smallest_dist) == x.shape[0]

    elif metric == 'l1':

        dists = compute_l1(x, y)
        smallest_dist = np.min(dists, axis=1)
        assert len(smallest_dist) == x.shape[0]

    else:
        smallest_dist = np.zeros(len(x))

    return smallest_dist


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
    """
    Computes the frequency of unique labels
    :param labels: list of labels
    :param weighted: bool, whether to compute weights instead of counts
    :return: label_counts (either as weights or as counts)
    """

    labels = labels.flatten()
    counts = Counter(labels)

    if not weighted:

        # Counts for different labels in sorted order
        counts = dict(sorted(counts.items(), key=lambda x: x[0]))
        label_counts = {k+1: v for k, v in counts.items()}
        return label_counts

    else:

        # Return the frequencies of different labels
        label_sum = np.sum(list(counts.values()))
        label_counts = dict()
        for key in counts:
            label_counts[key+1] = counts[key] / label_sum
            label_counts[key+1] = label_counts[key+1].round(4)

        label_counts = dict(sorted(label_counts.items(), key=lambda x: x[0]))

        return label_counts


def compute_wasserstein(real_data, real_labels, fake_data, expert_labels,
                        num_subpopulations, num_experts):

    wass_sums = list()

    for expert in range(num_experts):

        wass_sum_per_expert = list()
        expert_indices = np.where(expert_labels == expert)[0]

        if len(expert_indices) == 0:
            wass_sums.append([0] * num_subpopulations)  # TODO: What to add here?
            continue

        else:
            fake_data_by_expert = fake_data[expert_indices, :]

            for sub in range(num_subpopulations):
                subs_indices = np.where(real_labels == sub)[0]

                if len(subs_indices) == 0:
                    wass_sum_per_expert.append(np.inf)
                    continue

                else:
                    real_data_by_sub = real_data[subs_indices]
                    wass_sum = np.sum([
                        wasserstein_distance(real_data_by_sub[:, marker],
                                             fake_data_by_expert[:, marker])
                        for marker in range(real_data.shape[-1])
                    ])

                    wass_sum_per_expert.append(wass_sum)

        wass_sums.append(wass_sum_per_expert)

    wass_sums = np.asarray(wass_sums)
    assert wass_sums.shape == (num_experts, num_subpopulations)

    return wass_sums


def compute_ks(real_data, real_labels, fake_data, expert_labels,
               num_subpopulations, num_experts):
    """
    Computes ks_sums for each expert with each subpopulation by adding tests over individual markers

    :param real_data: Real cells, each with marker profile
    :param real_labels: Subpopulation generating those cells
    :param fake_data: Fake cells, generated by expert, with marker profile
    :param expert_labels: Which expert generates which cells
    :param num_subpopulations: Number of subpopulations in the data
    :param num_experts: Number of experts
    :return: ks_sums
    """

    ks_sums = list()

    for expert in range(num_experts):

        ks_sum_per_expert = list()
        expert_indices = np.where(expert_labels == expert)[0]

        if len(expert_indices) == 0:
            ks_sums.append([0] * num_subpopulations) #TODO: What to add here?
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
                    ks_sum = np.sum([
                        ks_2samp(real_data_by_sub[:, marker],
                                 fake_data_by_expert[:, marker])[0]
                        for marker in range(real_data.shape[-1])
                    ])

                    ks_sum_per_expert.append(ks_sum)

        ks_sums.append(ks_sum_per_expert)

    ks_sums = np.asarray(ks_sums)
    assert ks_sums.shape == (num_experts, num_subpopulations)

    return ks_sums


def compute_mmd(x, y, kernel='rbf', sigma=0.01, biased=True):

    if kernel == 'rbf':

        gamma = 1 / (2 * (sigma ** 2))

        k_xx = np.exp(-gamma * compute_l2(x, x))
        k_xy = np.exp(-gamma * compute_l2(x, y))
        k_yy = np.exp(-gamma * compute_l2(y, y))

        if biased:
            mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

        else:
            m = k_xx.shape[0]
            n = k_yy.shape[0]

            mmd = ((k_xx.sum() - m) / (m * (m - 1))
                   + (k_yy.sum() - n) / (n * (n - 1))
                   - 2 * k_xy.mean())

    else:
        raise NotImplementedError('No other kernels are supported currently')

    return mmd


def assign_expert_to_subpopulation(real_data, real_labels, fake_data,
                                   expert_labels, num_subpopulations,
                                   num_experts):
    """
    Assigns each expert to the subpopulation it learns based on sum of KS_tests for each marker
    :param real_data: Real cells, each with marker profile
    :param real_labels: Subpopulation generating those cells
    :param fake_data: Fake cells, generated by expert, with marker profile
    :param expert_labels: Which expert generates which cells
    :param num_subpopulations: Number of subpopulations in the data
    :param num_experts: Number of experts
    :return: expert_assignments, an array of size num_experts
    """

    ks_sums = compute_ks(
        real_data=real_data,
        real_labels=real_labels,
        fake_data=fake_data,
        expert_labels=expert_labels,
        num_subpopulations=num_subpopulations,
        num_experts=num_experts)

    expert_assignments = np.argmin(ks_sums, axis=1)

    return expert_assignments


def compute_learnt_subpopulation_weights(expert_labels, expert_assignments,
                                         num_subpopulations):
    """
    Computes learnt subpopulation weights
    :param expert_labels: expert labels corresponding to fake data
    :param expert_assignments: expert assignment to subpopulations
    :param num_subpopulations: Number of subpopulations in the data
    :return: learnt_subpopulation_weights
    """

    expert_weights = compute_frequency(labels=expert_labels, weighted=True)
    learnt_subpopulation_weights = {
        subpopulation+1: 0.0
        for subpopulation in range(num_subpopulations)
    }

    for subpopulation in range(num_subpopulations):

        if subpopulation not in expert_assignments:
            continue
        else:
            which_experts = np.where(expert_assignments == subpopulation)[0]
            for expert in which_experts:
                try:
                    learnt_subpopulation_weights[
                        subpopulation+1] += expert_weights[expert]

                except KeyError:
                    pass

    for key in learnt_subpopulation_weights:
        learnt_subpopulation_weights[key] = np.round(
            learnt_subpopulation_weights[key], 4)

    learnt_subpopulation_weights = dict(
        sorted(learnt_subpopulation_weights.items(), key=lambda x: x[0]))

    return learnt_subpopulation_weights


def load_model(out_dir, session_obj):
    """Load CellGAN model """

    model_name = 'model.ckpt'
    hparams_file = os.path.join(out_dir, 'Hparams.txt')
    model_path = os.path.join(out_dir, model_name)

    with open(hparams_file, 'r') as f:
        hparams = json.load(f)

    model = CellGan(
        noise_size=hparams['noise_size'],
        moe_sizes=hparams['moe_sizes'][1:-1],
        batch_size=hparams['batch_size'],
        num_markers=hparams['num_markers'],
        num_experts=hparams['num_experts'],
        g_filters=hparams['g_filters'],
        d_filters=np.array(hparams['d_filters']),
        d_pooled=np.array(hparams['d_pooled']),
        coeff_l1=hparams['coeff_l1'],
        coeff_l2=hparams['coeff_l2'],
        coeff_act=hparams['coeff_act'],
        num_top=hparams['num_top'],
        dropout_prob=hparams['dropout_prob'],
        noisy_gating=hparams['noisy_gating'],
        noise_eps=hparams['noise_eps'],
        d_lr=hparams['d_lr'],
        g_lr=hparams['g_lr'],
        beta_1=hparams['beta_1'],
        beta_2=hparams['beta_2'],
        reg_lambda=hparams['reg_lambda'],
        clip_val=hparams['clip_val'],
        train=True,
        init_method=hparams['init_method'],
        type_gan=hparams['type_gan'],
        load_balancing=hparams['load_balancing']
    )

    saver = tf.train.Saver()
    print("Loading Model")
    saver.restore(session_obj, model_path)
    print("Model Loaded")

    return model


def compute_f_measure(y_true, y_pred):
    """
    Compute f-measure of subpopulation prediction results.
    :param y_true:
    :param y_pred:
    :return: double, f-measure
    """

    y_true_unique = np.unique(y_true)
    y_pred_unique = np.unique(y_pred)

    N = len(y_true)
    f_measure_i = list()

    for i, y_i in enumerate(y_true_unique):
        f_measure_j = list()
        temp_ind_y = np.where(np.asarray(y_true) == y_i)[0]

        binary_y_i = np.zeros((N, ))
        binary_y_i[temp_ind_y] = 1

        n_c_i = len(temp_ind_y)
        for j, y_j in enumerate(y_pred_unique):
            temp_ind_y_j = np.where(np.asarray(y_pred) == y_j)[0]

            binary_y_j = np.zeros((N,))
            binary_y_j[temp_ind_y_j] = 1

            f_measure_j.append(f1_score(binary_y_i, binary_y_j))

        ind_max = np.argmax(np.asarray(f_measure_j))
        f_measure_i.append(n_c_i/N*f_measure_j[ind_max])

    return np.sum(f_measure_i)
