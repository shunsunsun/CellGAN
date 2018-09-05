
import numpy as np
import tensorflow as tf
from utils import build_real_data, sample_z
from utils import pca_plot, generate_random_subset
from utils import writeHparamsToFile, saveLossPlot
from utils import getFilters, getNpooled
# from utils import getBatches
from model import CellGan
import os
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###################################################################
# Test experiments on working models
###################################################################

# ---------------------------------------------
# Make the output directory if it doesn't exist
# ---------------------------------------------

test = 2
hard_drive = sys.path[0]
if test < 10:
    output_dir = hard_drive + '/Week 8/' + 'output_test_0' + str(test) + '/'
else:
    output_dir = hard_drive + '/Week 8/' + 'output_test_' + str(test) + '/'

experiment_name = 'Same Weights as _A05.fcs, no load balancing loss.'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------
# Real data parameters
# --------------------

n_subpopulations = 14

weights_subpopulations = np.array([0.007, 0.110, 0.026, 0.375, 0.118, 0.012, 0.01, 0.005,
                                   0.179, 0.002, 0.019, 0.125, 0.007, 0.005])

weights_subpopulations = weights_subpopulations/weights_subpopulations.sum()
n_markers = 10
n_cells_total = 40000

# ------------------
# CellGan parameters
# ------------------

noise_size = 100
batch_size = 100
ncell = 100

moe_sizes = [100, 100]
num_experts = 14
n_top = 1

nCellCnns = 30
gfilter = 20
maxFilter = 10
minFilter = 7

dfilters = getFilters(nCellCnns=nCellCnns, low=minFilter, high=maxFilter)
npooled = getNpooled(nfilters=nCellCnns, ncell=ncell)

learning_rate = 2e-4
train = True

# GAN and Optimizer Parameters

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

# -------------
# Get Real Data
# -------------

real_data, real_y_sub = build_real_data(
    n_sub=n_subpopulations,
    n_cells=n_cells_total,
    n_mark=n_markers,
    weight_sub=weights_subpopulations
)

# -----------------
# Build the CellGan
# -----------------

model = CellGan(noise_size=noise_size, batch_size=batch_size,
                moe_sizes=moe_sizes, num_experts=num_experts,
                ncell=ncell, nmark=n_markers, gfilter=gfilter,
                dfilters=dfilters, lr=learning_rate,
                train=train, beta2=beta2, n_top=n_top, npooled=npooled,
                reg_lambda=reg_lambda, beta1=beta1, typeGAN=typeGAN)

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
    'batch_size': batch_size
}

# Add flag for reinitialize

# -----------------------------------
# Logger Properties
# -----------------------------------

output_log = output_dir + 'Output.log'
logger = logging.getLogger('CellGan')
logging_format = '%(message)s'
handler = logging.FileHandler(output_log, mode='w')
formatter = logging.Formatter(logging_format)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

logger.info("Experiment Name: " + experiment_name + '\n')

logger.info("Starting our experiments "
            "with {} subpopulations \n".format(n_subpopulations))
logger.info("Number of filters "
            "in the CellCnn Ensemble are: {}".format(dfilters))
logger.info("number of Cells Pooled in "
            "the CellCnn Ensemble are: {} \n".format(npooled))
logger.info("The subpopulation weights are {} \n".format(weights_subpopulations))


# ----------------
# Training the GAN
# ----------------

D_loss = list()
G_loss = list()


# n_batches = 10
# train_batches = getBatches(inputs=real_data, batch_size=batch_size,
#                            ncell=ncell, n_batches=n_batches)


def getNextBatch(index):

    global train_batches

    return train_batches[index]


num_iter = 10000

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    writeHparamsToFile(
        dir_output=output_dir,
        hparams=hparams,
        run=test
    )

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

        # test_data, _ = generate_random_subset(
        #     inputs=real_data,
        #     ncell=ncell,
        #     batch_size=batch_size
        # )

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

            # print("Number of experts used: {}\n".format(experts_used))

            logger.info("Number of experts used: {} \n".format(experts_used))

            # Real Data
            # ---------

            test_real, indices = generate_random_subset(inputs=real_data,
                                                ncell=n_samples,
                                                batch_size=1)

            test_real = test_real.reshape(n_samples, n_markers)
            indices = np.reshape(indices, test_real.shape[0])
            subs_real = real_y_sub[indices]

            pca_plot(
                dir_output=output_dir,
                real_data=test_real,
                fake_data=test_fake,
                experts=experts,
                subs_real=subs_real,
                it=it
            )

            saveLossPlot(
                dir_output=output_dir,
                disc_loss=D_loss,
                gen_loss=G_loss
            )

            model_path = output_dir + 'model' + str(test) + '.ckpt'

            saver = tf.train.Saver()
            save_path = saver.save(sess, model_path)
