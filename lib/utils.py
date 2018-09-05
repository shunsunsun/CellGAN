
import tensorflow as tf
import numpy as np
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

xav_init = tf.contrib.layers.xavier_initializer
normal_init = tf.truncated_normal_initializer
zero_init = tf.zeros_initializer

initializers = dict()
initializers['xavier'] = xav_init
initializers['normal'] = normal_init
initializers['zeros'] = zero_init


def getFilters(nCellCnns, low, high):

    low = low
    high = high

    filters = np.random.randint(low=low, high=high, size=nCellCnns)

    return filters


def getNpooled(nfilters, ncell):

    nPooled = [np.random.randint(low=1, high=ncell) for _ in range(nfilters)]

    return nPooled


def sample_z(shape):

    noise = list()

    for _ in range(shape[0]):

        noise.append(np.random.multivariate_normal(
            mean=np.zeros(shape=shape[-1]),
            cov=np.eye(shape[-1]),
            size=shape[1]))

    return np.array(noise)


def build_real_data(n_sub, n_cells, weight_sub, n_mark):

    means = list()
    sd = list()

    for i in range(n_sub):
        means.append(np.random.choice(range(1, 5), n_mark, replace=True))
        sd.append(np.random.sample(n_mark))

    data = list()
    y_subpop = list()
    for i in range(n_sub):
        temp_n_cells = int(n_cells * weight_sub[i])
        data.append(np.random.normal(means[i], sd[i], (temp_n_cells, n_mark)))
        y_subpop.append([i + 1] * temp_n_cells)
    data = np.vstack(data)
    y_subpop = np.concatenate(y_subpop)

    # ind_shuffle = np.random.choice(range(n_cells), n_cells, replace=False)
    # data = data[ind_shuffle, :]
    # y_subpop = y_subpop[ind_shuffle]

    return data, y_subpop


def generate_random_subset(inputs, ncell, batch_size):

    shape = inputs.shape[0]
    indices = [np.random.choice(range(shape), ncell, replace=False)
               for i in range(batch_size)]

    subset = [inputs[index, ] for index in indices]

    return np.array(subset), indices


def getBatches(inputs, batch_size, n_batches, ncell):

    batches = [generate_random_subset(inputs=inputs, ncell=ncell, batch_size=batch_size)
               for _ in range(n_batches)]
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


def writeHparamsToFile(out_dir, hparams):

    filename = os.path.join(out_dir, 'Hparams.txt')

    with open(filename, 'w') as f:

        f.write(json.dumps(hparams))


def saveLossPlot(dir_output, disc_loss, gen_loss):

    filename = os.path.join(dir_output, '_loss_plot.jpg')
    plt.plot(range(len(disc_loss)), disc_loss, 'r', label='Discriminator Loss')
    plt.plot(range(len(gen_loss)), gen_loss, 'b', label='Generator Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()
