
import numpy as np
import tensorflow as tf
from lib.utils import build_gaussian_training_set, sample_z
from lib.utils import pca_plot, generate_random_subset
from lib.utils import save_loss_plot, write_hparams_to_file
from lib.utils import get_filters, get_num_pooled
from lib.model import CellGan
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

num_subpopulations = 14

weights_subpopulations = np.array([0.007, 0.110, 0.026, 0.375, 0.118, 0.012, 0.01, 0.005,
                                   0.179, 0.002, 0.019, 0.125, 0.007, 0.005])

weights_subpopulations = weights_subpopulations/weights_subpopulations.sum()
num_markers = 10
num_cells_total = 40000

# ------------------
# CellGan parameters
# ------------------

noise_size = 100
batch_size = 100
num_cells_per_input = 100

moe_sizes = [100, 100]
num_experts = 14
num_top = 1

num_cell_cnns = 30
num_filter_generator = 20
max_num_filter = 10
min_num_filter = 7

disc_filters = get_filters(num_cell_cnns=num_cell_cnns, low=min_num_filter, high=max_num_filter)
disc_pooled = get_num_pooled(num_cell_cnns=num_cell_cnns, num_cells_per_input=num_cells_per_input)

learning_rate = 2e-4
train = True

# GAN and Optimizer Parameters

beta_1 = 0.9
beta_2 = 0.999
reg_lambda = 5
type_gan = 'wgan'

if type_gan == 'wgan-gp':
    n_critic = 3
elif type_gan == 'wgan':
    n_critic = 2
else:
    n_critic = 1

# -------------
# Get Real Data
# -------------

real_data, real_y_sub = build_gaussian_training_set(
    num_subpopulations=num_subpopulations,
    num_cells=num_cells_total,
    num_markers=num_markers,
    weights_subpopulations=weights_subpopulations
)

# -----------------
# Build the CellGan
# -----------------

model = CellGan(noise_size=noise_size, moe_sizes=moe_sizes, batch_size=batch_size,
                num_markers=num_markers, d_pooled=disc_pooled, num_experts=num_experts, g_filters=num_filter_generator,
                d_filters=disc_filters, num_top=num_top, type_gan=type_gan, reg_lambda=reg_lambda,
                beta_1=beta_1, beta_2=beta_2, init_method='xavier')

# ----------------------------------
# Write hyperparameters to text file
# ----------------------------------

moe_in_size = model.generator.get_moe_input_size()

hparams = {
    'noise_size': noise_size,
    'num_experts': num_experts,
    'n_top': num_top,
    'learning_rate': learning_rate,
    'moe_sizes': [moe_in_size] + moe_sizes + [num_markers],
    'gfilter': num_filter_generator,
    'beta_1': beta_1,
    'beta_2': beta_2,
    'reg_lambda': reg_lambda,
    'n_critic': n_critic,
    'num_cell_per_input': num_cells_per_input,
    'num_cell_cnns': num_cell_cnns,
    'type_gan': type_gan,
    'min_num_filter': min_num_filter,
    'max_num_filter': min_num_filter,
    'batch_size': batch_size
}


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
            "with {} subpopulations \n".format(num_subpopulations))
logger.info("Number of filters "
            "in the CellCnn Ensemble are: {}".format(disc_filters))
logger.info("number of Cells Pooled in "
            "the CellCnn Ensemble are: {} \n".format(disc_pooled))
logger.info("The subpopulation weights are {} \n".format(weights_subpopulations))


# ----------------
# Training the GAN
# ----------------

discriminator_loss = list()
generator_loss = list()

num_iter = 10000

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    write_hparams_to_file(
        out_dir=output_dir,
        hparams=hparams,
    )

    for it in range(num_iter):

        # ----------------------
        # DISCRIMINATOR TRAINING
        # ----------------------

        model.set_train(True)

        for _ in range(n_critic):

            train_real, _ = generate_random_subset(
                inputs=real_data,
                num_cells_per_input=num_cells_per_input,
                batch_size=batch_size
            )

            z_shape = [batch_size, num_cells_per_input, noise_size]
            noise = sample_z(batch_size=batch_size, num_cells_per_input=num_cells_per_input,
                             noise_size=noise_size)

            if model.hparams['type_gan'] == 'wgan':

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

        noise = sample_z(batch_size=batch_size, num_cells_per_input=num_cells_per_input,
                         noise_size=noise_size)

        fetches = [model.g_solver, model.g_loss, model.generator.moe_loss]

        feed_dict = {
            model.Z: noise,
            model.X: train_real
        }

        _, g_loss, moe_loss = sess.run(
            fetches, feed_dict=feed_dict
        )

        discriminator_loss.append(d_loss)
        generator_loss.append(g_loss)

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

            num_samples = 1000
            input_noise = sample_z(batch_size=1, num_cells_per_input=num_samples,
                                   noise_size=noise_size)

            fetches = [model.g_sample, model.generator.gates]
            feed_dict = {
                model.Z: input_noise
            }

            test_fake, gates = sess.run(
                fetches=fetches,
                feed_dict=feed_dict
            )

            test_fake = test_fake.reshape(num_samples, num_markers)

            experts = np.argmax(gates, axis=1)
            experts_used = len(np.unique(experts))

            # print("Number of experts used: {}\n".format(experts_used))

            logger.info("Number of experts used: {} \n".format(experts_used))

            # Real Data
            # ---------

            test_real, indices = generate_random_subset(inputs=real_data, num_cells_per_input=num_samples,
                                                        batch_size=1)

            test_real = test_real.reshape(num_samples, num_markers)
            indices = np.reshape(indices, test_real.shape[0])
            real_subs = real_y_sub[indices]

            # pca_plot(out_dir=output_dir, real_data=test_real, fake_data=test_fake,
            #         experts=experts, real_subs=real_subs, it=it)

            save_loss_plot(out_dir=output_dir, disc_loss=discriminator_loss,
                           gen_loss=generator_loss)

            model_path = output_dir + 'model' + str(test) + '.ckpt'

            saver = tf.train.Saver()
            save_path = saver.save(sess, model_path)
