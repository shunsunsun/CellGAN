
import tensorflow as tf
import numpy as np
import os
import json

import matplotlib.pyplot as plt

xav_init = tf.contrib.layers.xavier_initializer
normal_init = tf.truncated_normal_initializer
zero_init = tf.zeros_initializer

# Different methods for initializing the data
initializers = dict()
initializers['xavier'] = xav_init
initializers['normal'] = normal_init
initializers['zeros'] = zero_init


def f_trans(x, c):
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

    batches = [generate_random_subset(inputs=inputs, num_cells_per_input=num_cells_per_input,
                                      batch_size=batch_size, weights=weights, return_indices=False)
               for _ in range(num_batches)]
    return batches


def generate_random_subset(inputs, num_cells_per_input, batch_size, weights=None, return_indices=False):

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


def save_loss_plot(out_dir, disc_loss, gen_loss):

    """
    Saves loss plot to output directory
    :param out_dir: str, output directory
    :param disc_loss: list, discriminator losses
    :param gen_loss: list, generator losses
    :return: no returns
    """

    filename = os.path.join(out_dir, 'loss_plot.jpg')
    plt.plot(range(len(disc_loss)), disc_loss, 'r', label='Discriminator Loss')
    plt.plot(range(len(gen_loss)), gen_loss, 'b', label='Generator Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()
