
import tensorflow as tf
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

xav_init = tf.contrib.layers.xavier_initializer
normal_init = tf.truncated_normal_initializer
zero_init = tf.zeros_initializer

# Different methods for initializing the data
initializers = dict()
initializers['xavier'] = xav_init
initializers['normal'] = normal_init
initializers['zeros'] = zero_init


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


def generate_random_subset(inputs, num_cells_per_input, batch_size):

    """
    Returns a random subset from input data of shape (batch_size, num_cells_per_input, num_markers)
    :param inputs: numpy array, the input ndarray to sample from
    :param num_cells_per_input: int, number of cells per multi-cell input
    :param batch_size: int, batch size of the subset
    :return:
    """

    num_cells_total = inputs.shape[0]
    indices = [np.random.choice(range(num_cells_total), num_cells_per_input, replace=False)
               for i in range(batch_size)]

    subset = [inputs[index, ] for index in indices]

    return np.array(subset), indices


def get_batches(inputs, batch_size, num_batches, num_cells_per_input):

    """
    Generate multiple batches of training data from given inputs
    :param inputs: input numpy array of shape (num_cells, num_markers)
    :param batch_size: int, batch_size of each batch generated
    :param num_batches: int, number of batches of training data to generate
    :param num_cells_per_input: int, number of cells per multi-cell input
    :return:
    """

    batches = [generate_random_subset(inputs=inputs, num_cells_per_input=num_cells_per_input, batch_size=batch_size)
               for _ in range(num_batches)]
    return batches


def pca_plot(out_dir, real_data, fake_data, it, real_subs, experts):

    subpops = np.unique(real_subs)
    markers = np.arange(real_data.shape[-1])

    ncol = markers.shape[0]
    nrow = len(subpops)

    dirname = os.path.join(out_dir, str((it//100)+1))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    unique_experts = np.unique(experts)

    for expert in unique_experts:

        filename = os.path.join(dirname, 'Expert_' + str(expert + 1) + '.png')
        expert_data = fake_data[np.where(experts == expert)[0], :]

        f, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(25, 25))
        best_sub = None
        best_KS_sum = np.inf

        for sub in subpops:

            KS_markers = list()

            indices = np.where(real_subs == sub)[0]
            sub_data = real_data[indices, :]

            for marker in markers:

                KS = ks_2samp(sub_data[:, marker], expert_data[:, marker])[0]
                KS_markers.append(KS)

                bins = np.linspace(sub_data[:, marker].min(), sub_data[:, marker].max(), num=30)
                ax[sub, marker].hist(x=sub_data[:, marker], bins=bins, label='R', alpha=0.5)
                ax[sub, marker].hist(x=expert_data[:, marker], bins=bins, label='F', alpha=0.5)
                ax[sub, marker].set_title('Sub {0}, Mark {1}, KS {2:.2f}'.format(sub + 1, marker + 1, KS))
                ax[sub, marker].legend()

            if np.sum(KS_markers) <= best_KS_sum:
                best_KS_sum = np.sum(KS_markers)
                best_sub = sub

        for marker in markers:

            ax[best_sub, marker].spines['bottom'].set_color('0.0')
            ax[best_sub, marker].spines['top'].set_color('0.0')
            ax[best_sub, marker].spines['right'].set_color('0.0')
            ax[best_sub, marker].spines['left'].set_color('0.0')
            [i.set_linewidth(2.5) for i in ax[best_sub, marker].spines.values()]

        f.tight_layout()
        plt.savefig(filename)
        plt.close()


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
