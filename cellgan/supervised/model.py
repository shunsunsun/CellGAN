import numpy as np
import tensorflow as tf
import json
import os

from cellgan.supervised.defaults import *
from cellgan.lib.utils import sample_z, generate_subset, load_model
from cellgan.lib.utils import initializers, compute_outlier_weights
from cellgan.lib.utils import compute_f_measure_uniformly_weighted
from scipy.cluster.hierarchy import fcluster, linkage

DEFAULT_HIDDEN_UNITS = 256

class Trainer(object):
    """
    Trains a supervised model on data generated by GAN to predict classes
    Predicts classes for real data 
    """

    def __init__(self, exp_name, iteration, sess_obj, num_samples=10000, mb_size=64, num_filters=20, 
        num_pooled=20, lr=1e-4, dropout_prob=0.4, beta1=0.9, beta2=0.999, epsilon=1e-8, init_method='xavier', 
        inhibitor='AKTi', sample='outlier'):
        self.exp_name = exp_name
        self.model_iter = iteration
        self.sess_obj = sess_obj
        self.batch_size = mb_size
        self.num_samples = num_samples
        self.num_filters = num_filters
        self.num_pooled = num_pooled
        self.dropout_prob = dropout_prob
        self.inhibitor = inhibitor
        self.sample_method = sample
        self.init = initializers[init_method]

        self.exp_hparams = self.load_hparams()
        self.noise_size = self.exp_hparams["noise_size"]
        self.num_markers = self.exp_hparams["num_markers"]
        self.num_classes = self.exp_hparams["num_experts"]
        self.nc_input = self.exp_hparams["num_cell_per_input"]

        # Optimizer parameters
        self.opt_params = dict()
        self.opt_params["lr"] = lr
        self.opt_params["beta1"] = beta1
        self.opt_params["beta2"] = beta2
        self.opt_params["epsilon"] = epsilon

        self._prepare_training_data()
        self._compute_loss()
        self._solvers()

    def load_hparams(self):
        hparams_file = os.path.join(RESULTS_DIR, self.inhibitor, self.exp_name, 'Hparams.txt')
        with open(hparams_file, 'r') as f:
            hparams = json.load(f)
        return hparams

    def _generate_samples(self, num_samples=10000):
        """Generates data from the GAN model."""
        noise_sample = sample_z(batch_size=1, num_cells_per_input=num_samples, noise_size=self.noise_size)
        fetches = [self.model.g_sample, self.model.generator.gates]
        feed_dict = {self.model.Z: noise_sample}

        fake_samples, gates = self.sess_obj.run(fetches=fetches, feed_dict=feed_dict)
        fake_samples = fake_samples.reshape(num_samples, self.num_markers)
        fake_sample_experts = np.argmax(gates, axis=1)

        return fake_samples, fake_sample_experts

    def _prepare_training_data(self):
        """Prepares training data, by sampling from the GAN model."""
        out_dir = os.path.join(RESULTS_DIR, self.inhibitor, self.exp_name)
        self.model = load_model(out_dir=out_dir, session_obj=self.sess_obj, iteration=self.model_iter)
        noise_sample = sample_z(batch_size=1, num_cells_per_input=self.num_samples, noise_size=self.noise_size)

        fetches = [self.model.g_sample, self.model.generator.gates, self.model.generator.logits]
        feed_dict = {self.model.Z: noise_sample}

        fake_samples, gates, logits = self.sess_obj.run(fetches=fetches, feed_dict=feed_dict)
        fake_samples = fake_samples.reshape(self.num_samples, self.num_markers)
        fake_sample_experts = np.argmax(gates, axis=1)

        self.training_data = fake_samples
        self.training_labels = fake_sample_experts

    def _setup_placeholders(self):
        self.X = tf.placeholder(name='X', shape=[None, None, self.num_markers], dtype=tf.float32)
        self.labels = tf.placeholder(name='labels', shape=[None, None, 1], dtype=tf.int32)

    def _build_graph(self, reuse=tf.AUTO_REUSE, print_shape=False):
        """Setup the computation graph """

        with tf.variable_scope("Trainer", reuse=reuse):
            batch_size = tf.shape(self.X)[0]
            num_cells_per_input = tf.shape(self.X)[1]
            num_markers = int(self.X.shape[-1])

            conv1_input = tf.reshape(self.X, shape=[batch_size * num_cells_per_input, num_markers, 1])

            # Expected shape: (batch_size*num_cells_per_input, 1, num_filters)
            d_conv1 = tf.layers.conv1d(
                inputs=conv1_input,
                filters=self.num_filters,
                kernel_size=num_markers,
                kernel_initializer=self.init(),
                activation=tf.nn.relu,
                name='d_conv1',
            )

            reshaped_dconv1 = tf.reshape(d_conv1, shape=[batch_size, num_cells_per_input, self.num_filters])

            # Expected Shape: (batch_size*num_pooled, DEFAULT_HIDDEN_UNITS)

            d_dense1 = tf.layers.dense(
                inputs=reshaped_dconv1,
                units=DEFAULT_HIDDEN_UNITS,
                name="d_dense1",
                activation=tf.nn.relu)

            # Expected Shape: (batch_size*num_pooled, DEFAULT_HIDDEN_UNITS)
            d_dropout1 = tf.layers.dropout(
                inputs=d_dense1,
                rate=self.dropout_prob,
                name="d_dropout1")

            # Expected Shape = (batch_size*num_pooled, 1)
            self.d_dense2 = tf.layers.dense(
                inputs=d_dropout1, units=self.num_classes, name='d_dense2', activation=None)

            if print_shape:
                print()
                print("Discriminator")
                print("-------------")
                print("Convolutional layer input shape: ", conv1_input.shape)
                print("Convolutional layer output shape: ", d_conv1.shape)
                print("Convolutional layer output reshaped: ",
                      reshaped_dconv1.shape)
                print("Dense Layer 1 output shape: ", d_dense1.shape)
                print("Dense Layer 2 output shape: ", self.d_dense2.shape)
                print()

    def _compute_loss(self):
        """Computes the cross entropy loss."""
        self._setup_placeholders()
        self._build_graph()

        one_hot_labels = tf.one_hot(self.labels, depth=self.num_classes, axis=-1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.d_dense2, labels=one_hot_labels))

    def _solvers(self):
        """Defines the optimizer to be used."""
        with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.opt_params["lr"],
                beta1=self.opt_params["beta1"],
                beta2=self.opt_params["beta2"],
                epsilon=self.opt_params["epsilon"])

            self.params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Trainer")
            self.solver = optimizer.minimize(loss=self.loss, var_list=self.params)

    def compute_f_score(self, X, y, nruns=10):
        """Computes the F-Measure during training."""

        f_measures = list()
        num_experts = self.exp_hparams['num_experts']
        preds = self.predict(X)

        for run in range(nruns):
            fake_samples, fake_sample_experts = self._generate_samples(num_samples=10000)
            means = list()
            for expert in range(num_experts):
                indices = np.flatnonzero(fake_sample_experts == expert)
                means.append(np.mean(fake_samples[indices], axis=0))
            means = np.array(means)

            Z = linkage(means, 'ward')

            f_scores = list()
            for height in range(num_experts):
                clusters = fcluster(Z, height, criterion='distance')

                cluster_labels = list()
                for i in range(len(X)):
                    cluster_labels.append(clusters[preds[i]])
                f_scores.append(compute_f_measure_uniformly_weighted(y, cluster_labels))

            f_measures.append(max(f_scores))

        return np.mean(f_measures), np.std(f_measures)


    def fit(self, X, y, num_iterations=1, print_every_n=1, nruns=10):
        """ Fits the model """
        losses = list()
        mean_fs = list()
        std_fs = list()

        for iteration in range(num_iterations):
            fetches = [self.loss, self.solver]

            if self.sample_method == 'outlier':
                subset_size = np.random.randint(low=20, high=50)
                outlier_scores = compute_outlier_weights(
                    inputs=self.training_data, method='q_sp', subset_size=subset_size)
                
                train_batch, indices = generate_subset(
                    inputs=self.training_data, 
                    num_cells_per_input=self.nc_input, 
                    batch_size=self.batch_size, 
                    weights=outlier_scores, 
                    return_indices=True)
            else:
                train_batch, indices = generate_subset(
                    inputs=self.training_data, 
                    num_cells_per_input=self.nc_input, 
                    batch_size=self.batch_size, 
                    weights=None, 
                    return_indices=True)

            train_batch_labels = self.training_labels[indices]
            train_batch_labels = np.reshape(train_batch_labels, (self.batch_size, self.nc_input, -1))
            feed_dict = {self.X: train_batch, self.labels: train_batch_labels}

            loss, _ = self.sess_obj.run(fetches=fetches, feed_dict=feed_dict)
            
            if iteration % print_every_n == 0:
                mean_f, std_f = self.compute_f_score(X, y, nruns=nruns)
                print("Iteration: {0:1d}, Loss: {1:.3f}, F-measure: {2:.3f} +/- {3:.3f}".format(iteration, loss, mean_f, std_f))
                losses.append(loss)
                mean_fs.append(mean_f)
                std_fs.append(std_f)

        return losses, mean_fs, std_fs


    def predict(self, X):
        """Generate predictions for given sample X."""
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]

        fetches = self.d_dense2
        feed_dict = {self.X: X}

        logits = self.sess_obj.run(fetches, feed_dict)
        logits = np.squeeze(logits, axis=0)
        preds = np.argmax(logits, axis=1)

        return preds


if __name__ == '__main__':
    exp_name = "21_02_2019-22_12_26"
    iteration = 6001

    with tf.Session() as sess:
        model = Trainer(exp_name, iteration, sess_obj=sess)
