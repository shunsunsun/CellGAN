
import tensorflow as tf
from Generator import CellGan_gen
from Ensemble import CellCnn_Ensemble


class CellGan(object):

    """Creates an object of type CellGan

    Args:
        - noise_size: int
            Size of the input noise to the generator

        - moe_sizes: list of ints
            Size of the hidden layer used for the MoE

        - num_experts: int, default 3
            Number of experts used in the CellGan_generator

        - ncell: int, default 200
            Number of cells per multi-cell input

        - nmark: int, default 10
            Number of markers used in the experiment

        - gfilter: int, default 10
            Number of filters used in CellGan_gen

        - dfilters: list, default [10]
            Number of filters used in CellCnn_Ensemble

        - coeff_l1: float, default 0
            Coefficient of the l1 regularizer

        - coeff_l2: float, default 1e-4
            Coefficient of the l2 regularizer

        - coeff_act: float, default 0
            Coefficient of the activity regularizer

        - npooled: list, default [3]
            Number of cells pooled in max_pooling for CellCnn_Ensemble

        - dropout: str, default 'auto'
            Whether dropout is used

        - dropout_p: float, default 0.5
            dropout rate

        - lr: float, default 1e-4
            learning rate used for the optimizers

        - init_method: str, default 'xavier'
            Method for initializing weights

        - typeGAN: str, default 'Normal'
            Type of GAN used

        - beta1: float, default 0.9
            beta1 value for Adam Optimizer

        - beta2: float, default 0.999
            beta2 value for Adam Optimizer

        - reg_lambda: float, default 10
            gradient penalty regularizer for WGAN

    """

    def __init__(self, noise_size, moe_sizes, batch_size, num_experts=3,
                 ncell=200, nmark=10, gfilter=10, dfilters=[10], coeff_l1=0,
                 coeff_l2=1e-4, coeff_act=0, npooled=[3], dropout_p=0.4,
                 lr=1e-4, n_top=1, noisy_gating=True, noise_eps=1e-2,
                 train=False, init_method='xavier', typeGAN='Normal',
                 beta1=0.9, beta2=0.999, reg_lambda=10, load_balancing=False):

        self.generator = CellGan_gen(moe_sizes=moe_sizes, ncell=ncell,
                                     nmark=nmark, num_experts=num_experts,
                                     init_method=init_method, n_top=n_top,
                                     noisy_gating=noisy_gating,
                                     noise_epsilon=noise_eps, gfilter=gfilter)

        self.discriminator = CellCnn_Ensemble(ncell=ncell, nmark=nmark,
                                              nfilters=dfilters, coeff_l1=coeff_l1,
                                              coeff_l2=coeff_l2, coeff_act=coeff_act,
                                              npooled=npooled, init_method=init_method,
                                              dropout_p=dropout_p)
        self.nDiscriminator = len(dfilters)

        self.noise_size = noise_size
        self.batch_size = batch_size
        self.typeGAN = typeGAN
        self.lr = lr
        self.load_balancing = load_balancing

        self.g_sample = None
        self.d_real = None
        self.d_fake = None

        self.reg_lambda = reg_lambda
        self.train = train

        self.AdOptim = dict()
        self.AdOptim['beta1'] = beta1
        self.AdOptim['beta2'] = beta2

        self._create_placeholders()
        self._compute_loss()
        self._solvers()

        self.reuse = tf.AUTO_REUSE

    def set_train(self, train):

        """
        Set the train value

        Inputs
        ------

        train:      bool, True if training, False otherwise
                    - Controls noise addition in the gating network

        """

        self.train = train

    def _create_placeholders(self):

        """
        Creates the placeholders Z and X for the input
        noise and real samples

        """

        nmark = self.discriminator.input_features['nmark']

        self.Z = tf.placeholder(
            shape=[None, None, self.noise_size],
            dtype=tf.float32,
            name="input_noise")

        self.X = tf.placeholder(
            shape=[None, None, nmark],
            dtype=tf.float32,
            name="Real_samples")

        self.shape = tf.shape(self.X)

    def _generate_sample(self, reuse=tf.AUTO_REUSE):

        """
        For a given input, produces a fake sample from the generator

        Inputs
        ------

        reuse:      tensorflow Reuse Object, default tf.AUTO_REUSE
                    -Indicates whether the variables can be used

        Returns
        -------

        output:     tensor, shape: (batch_size, ncell, nmark)
                    - The fake multi-cell input

        """
        output = self.generator.build_gen(inputs=self.Z,
                                          train=self.train,
                                          reuse=reuse)

        return output

    def _eval_disc(self, inputs, reuse):

        """
        Produces the output of the ensemble of CellCnns

        Inputs
        ------

        inputs:     tensor, (batch_size, ncell, nmark)
                    - Multi cell inputs

        reuse:      bool
                    - Indicates whether defined variables need
                    to be reused

        Returns
        -------

        outputs:    dict, keys are the CellCnn indices
                    - Contains the dictionary of outputs from
                    each CellCnn

        """
        output = self.discriminator._eval_ensemble(inputs=inputs,
                                                   reuse=reuse)

        return output

    # def _calc_grad(self):

    #    """ Calculates the gradient needed as part of the WGAN-GP Loss """

    #    self.grad = tf.gradients(
    #        ys=self._eval_disc(inputs=self.X_new),
    #        xs=[self.X_new])[0]

    #    self.grad_norm = tf.sqrt(tf.reduce_sum(self.grad**2, axis=1))

    #    return self.grad_norm

    def _compute_loss(self):

        """
        Calculates discriminator and generator loss for specified GAN

        """

        self.g_sample = self._generate_sample()
        self.d_real = self._eval_disc(inputs=self.X, reuse=tf.AUTO_REUSE)
        self.d_fake = self._eval_disc(inputs=self.g_sample, reuse=tf.AUTO_REUSE)

        if self.typeGAN == 'Normal':

            self.d_loss_real = 0

            for i in self.d_real:
                self.d_loss_real += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_real[i],
                        labels=tf.ones_like(self.d_real[i])))

            self.d_loss_fake = 0
            self.g_loss_fake = 0

            for i in self.d_fake:

                self.d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_fake[i],
                        labels=tf.zeros_like(self.d_fake[i])))

                self.g_loss_fake += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.d_fake[i],
                        labels=tf.ones_like(self.d_fake[i])))

            self.d_loss = self.d_loss_fake + self.d_loss_real
            self.g_loss = self.g_loss_fake

        elif self.typeGAN == 'WGAN':

            self.d_loss_fake = 0
            self.d_loss_real = 0

            for i in self.d_fake:
                self.d_loss_fake += tf.reduce_mean(self.d_fake[i])

            for i in self.d_real:
                self.d_loss_real += tf.reduce_mean(self.d_real[i])

            self.d_loss = self.d_loss_fake - self.d_loss_real
            chosen_experts = tf.argmax(self.generator.gates, axis=1)
            unique_experts, _ = tf.unique(chosen_experts)

            self.experts_used = tf.reduce_sum(
                tf.ones_like(unique_experts, dtype=tf.float32))

            if self.load_balancing:
                self.g_loss = -self.d_loss_fake + self.generator.moe_loss
            else:
                self.g_loss = -self.d_loss_fake

        self.d_params = tf.get_collection(
            key=tf.GraphKeys.GLOBAL_VARIABLES,
            scope="Ensemble")

        self.g_params = tf.get_collection(
            key=tf.GraphKeys.GLOBAL_VARIABLES,
            scope="CellGan_gen")

    def _solvers(self):

        """Builds the optimizers to be used for training the GAN """

        if self.typeGAN == 'WGAN':

            d_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            g_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)

            self.d_solver = d_opt.minimize(
                loss=self.d_loss,
                var_list=self.d_params)

            self.g_solver = g_opt.minimize(
                loss=self.g_loss,
                var_list=self.g_params)

            self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01))
                           for p in self.d_params]

        else:
            d_opt = tf.train.AdamOptimizer(
                learning_rate=self.lr,
                beta1=self.AdOptim['beta1'],
                beta2=self.AdOptim['beta2']
            )
            g_opt = tf.train.AdamOptimizer(
                learning_rate=self.lr,
                beta1=self.AdOptim['beta1'],
                beta2=self.AdOptim['beta2']
            )

            self.d_solver = d_opt.minimize(
                loss=self.d_loss,
                var_list=self.d_params)

            self.g_solver = g_opt.minimize(
                loss=self.g_loss,
                var_list=self.g_params)
