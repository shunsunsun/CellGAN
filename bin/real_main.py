
import numpy as np
import tensorflow as tf
from utils import sample_z
from utils import pca_plot, generate_random_subset
from utils import writeHparamsToFile, saveLossPlot
from utils import getFilters, getNpooled
from model import CellGan
import os
import sys
import logging
import pickle
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###################################################################
# Experiments on real data
###################################################################

# ------------------------------------
# Output Directory Creation and Checks
# ------------------------------------

inhibitor_used = 'Crassin'
regex_well = '_C12'

test = 2
hard_drive = os.path.join(sys.path[0], 'Real Data Tests')

if test < 10:
    regex_folder = regex_well + '_0' + str(test)
else:
    regex_folder = regex_well + '_' + str(test)

output_dir = os.path.join(hard_drive, inhibitor_used, regex_folder)

model_name = 'model' + regex_folder + '.ckpt'
model_path = os.path.join(output_dir, model_name)

if os.path.exists(output_dir):
    raise IsADirectoryError('Directory already exists. Change values accordingly')

data_dir = os.path.join(sys.path[0], 'Real Data')
data_file = os.path.join(data_dir, inhibitor_used, regex_well) + '.pkl'
label_file = os.path.join(data_dir, inhibitor_used, regex_well) + '_label.pkl'

if not os.path.exists(data_file):
    raise FileNotFoundError('File: ' + str(data_file.split('\\')[-1]) +
                            ' not found in the directory specified. Please add it')
else:
    print('File: ' + str(data_file.split('\\')[-1]) + ' exists')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------
# Loading Real Data
# --------------------

with open(data_file, 'rb') as f:
    real_data = pickle.load(f, encoding='latin1')

with open(label_file, 'rb') as f:
    real_subs = pickle.load(f, encoding='latin1')

n_markers = real_data.shape[-1]

# ------------------
# CellGan parameters
# ------------------

noise_size = 100
batch_size = 100
ncell = 100

moe_sizes = [100, 100]
num_experts = 9
n_top = 1

nCellCnns = 30
gfilter = 20
maxFilter = 10
minFilter = 7

dfilters = getFilters(nCellCnns=nCellCnns, low=minFilter, high=maxFilter)
npooled = getNpooled(nfilters=nCellCnns, ncell=ncell)

learning_rate = 2e-4
train = True
load_balancing = False

# Optimizer Parameters

beta1 = 0.9
beta2 = 0.999
reg_lambda = 5
typeGAN = 'WGAN'

if typeGAN == 'WGAN-GP':
    n_critic = 3
elif typeGAN == 'WGAN':
    n_critic = 2
else:
    n_critic = 1

# -----------------
# Build the CellGan
# -----------------

print('Building the CellGan...')

model = CellGan(noise_size=noise_size, batch_size=batch_size,
                moe_sizes=moe_sizes, num_experts=num_experts,
                ncell=ncell, nmark=n_markers, gfilter=gfilter,
                dfilters=dfilters, lr=learning_rate,
                train=train, beta2=beta2, n_top=n_top, npooled=npooled,
                reg_lambda=reg_lambda, beta1=beta1, typeGAN=typeGAN,
                load_balancing=load_balancing)

print('CellGan Built.')
# ----------------------------------
# Write hyperparameters to text file
# ----------------------------------

moe_in_size = model.generator.get_moe_input_size()

hparams = {
    'noise_size': noise_size,
    'num_experts': num_experts,
    'n_top': n_top,
    'learning_rate': learning_rate,
    'moe_sizes': [moe_in_size] + moe_sizes + [n_markers],
    'gfilter': gfilter,
    'beta1': beta1,
    'beta2': beta2,
    'reg_lambda': reg_lambda,
    'n_critic': n_critic,
    'ncell': ncell,
    'nCellCnns': nCellCnns,
    'typeGAN': typeGAN,
    'minFilter': maxFilter,
    'maxFilter': minFilter,
    'batch_size': batch_size,
    'nmark': n_markers,
    'dfilters': dfilters.tolist(),
    'npooled': npooled,
    'load_balancing': load_balancing
}

writeHparamsToFile(
    out_dir=output_dir,
    hparams=hparams,
)

# -----------------------------------
# Logger Properties
# -----------------------------------

output_log = os.path.join(output_dir, 'Output.log')
logger = logging.getLogger('CellGan')
logging_format = '%(message)s'
handler = logging.FileHandler(output_log, mode='w')
formatter = logging.Formatter(logging_format)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

logger.info("Inhibitor Used: {}".format(inhibitor_used))
logger.info("Stimulation Method: {}\n".format(regex_well))

if load_balancing:
    logger.info("Load Balancing Loss added as well. \n")

logger.info("Number of filters "
            "in the CellCnn Ensemble are: {}".format(dfilters.tolist()))
logger.info("number of Cells Pooled in "
            "the CellCnn Ensemble are: {} \n".format(npooled))


# ----------------
# Training the GAN
# ----------------

D_loss = list()
G_loss = list()

num_iter = 10000

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for it in range(num_iter):

        # ----------------------
        # DISCRIMINATOR TRAINING
        # ----------------------

        model.set_train(True)

        for _ in range(n_critic):

            train_real, _ = generate_random_subset(
                inputs=real_data,
                ncell=ncell,
                batch_size=batch_size
            )

            z_shape = [batch_size, ncell, noise_size]
            noise = sample_z(shape=z_shape)

            if model.typeGAN == 'WGAN':

                fetches = [
                    model.d_solver, model.d_loss,
                    model.clip_D
                ]

                feed_dict = {
                    model.Z: noise,
                    model.X: train_real
                }

                _, d_loss, _ = sess.run(
                    fetches,
                    feed_dict=feed_dict
                )

            else:

                fetches = [
                    model.d_solver, model.d_loss
                ]

                feed_dict = {
                    model.Z: noise,
                    model.X: train_real
                }

                _, d_loss = sess.run(
                    fetches,
                    feed_dict=feed_dict
                )

        # ------------------
        # GENERATOR TRAINING
        # ------------------

        z_shape = [batch_size, ncell, noise_size]
        noise = sample_z(shape=z_shape)

        fetches = [
            model.g_solver, model.g_loss, model.experts_used,
            model.generator.moe_loss
        ]

        feed_dict = {
            model.Z: noise,
            model.X: train_real
        }

        _, g_loss, ngates, moe_loss = sess.run(
            fetches, feed_dict=feed_dict
        )

        D_loss.append(d_loss)
        G_loss.append(g_loss)

        # ----------------------------------------
        # PCA Plotting of samples
        # ----------------------------------------

        if it % 100 == 0:

            model.set_train(False)

            print("We are at iteration: {}".format(it + 1))
            # print("Discriminator Loss: {}".format(d_loss))
            # print("Generator Loss: {}".format(g_loss))
            # print("Moe Loss: {}".format(moe_loss))

            logger.info("We are at iteration: {}".format(it + 1))
            logger.info("Discriminator Loss: {}".format(d_loss))
            logger.info("Generator Loss: {}".format(g_loss))
            logger.info("Load Balancing Loss: {} \n".format(moe_loss))

            # Fake Data
            # ---------

            n_samples = 1000
            input_noise = sample_z(shape=[1, n_samples, noise_size])

            fetches = [model.g_sample, model.generator.gates]
            feed_dict = {
                model.Z: input_noise
            }

            test_fake, gates = sess.run(
                fetches=fetches,
                feed_dict=feed_dict
            )

            test_fake = test_fake.reshape(n_samples, n_markers)

            experts = np.argmax(gates, axis=1)
            experts_used = len(np.unique(experts))

            # logger.info("Number of experts used: {}".format(experts_used))

            # Real Data
            # ---------

            test_real, indices = generate_random_subset(inputs=real_data,
                                                ncell=n_samples,
                                                batch_size=1)

            test_real = test_real.reshape(n_samples, n_markers)
            indices = np.reshape(indices, test_real.shape[0])
            test_real_subs = real_subs[indices]

            # This is just for understanding, not used anywhere in training the network

            logger.info("Number of subpopulations present: {} \n".format(len(np.unique(test_real_subs))))

            test_data_freq = dict(Counter(test_real_subs))
            total = sum(test_data_freq.values(), 0.0)
            test_data_freq = {k: round(test_data_freq[k]/total, 3) for k in sorted(test_data_freq.keys())}

            real_data_freq = dict(Counter(real_subs))
            total = sum(real_data_freq.values(), 0.0)
            real_data_freq = {k: round(real_data_freq[k]/total, 3) for k in sorted(real_data_freq.keys())}

            fake_data_freq = dict(Counter(experts))
            total = sum(fake_data_freq.values(), 0.0)
            fake_data_freq = {k: round(fake_data_freq[k]/total, 3) for k in sorted(fake_data_freq.keys())}

            logger.info("Frequency of subpopulations in real data is: {}".format(real_data_freq))
            logger.info("Frequency of subpopulations in test data is: {} \n".format(test_data_freq))

            logger.info("Checking Weights")
            logger.info("Real subpopulation weights: {}".format(sorted(real_data_freq.values(), reverse=True)))
            logger.info("Expert subpopulation weights: {}\n".format(sorted(fake_data_freq.values(), reverse=True)))

            logger.info("-----------\n")

            pca_plot(
                out_dir=output_dir,
                real_data=test_real,
                fake_data=test_fake,
                experts=experts,
                real_subs=test_real_subs,
                it=it
            )

            saveLossPlot(
                dir_output=output_dir,
                disc_loss=D_loss,
                gen_loss=G_loss
            )

            saver = tf.train.Saver()
            save_path = saver.save(sess, model_path)
