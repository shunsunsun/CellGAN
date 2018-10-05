import tensorflow as tf
from lib.generator import CellGanGen
from lib.ensemble import CellCnnEnsemble


class CellGan(object):
    """Creates an object of type CellGan

    Args:
        - noise_size: int
            Size of the input noise to the generator

        - moe_sizes: list of ints
            Size of the hidden layer used for the MoE

        - batch_size: int
            Batch size used for training

        - num_markers: int, default 10
            Number of markers whose profiles we want to learn

        - num_experts: int, default 3
            Number of experts used in the CellGan_generator

        - g_filters: int, default 10
            Number of filters used in CellGan_gen

        - d_filters: list, default [10]
            Number of filters used in CellCnn_Ensemble

        - coeff_l1: float, default 0
            Coefficient of the l1 regularizer

        - coeff_l2: float, default 1e-4
            Coefficient of the l2 regularizer

        - coeff_act: float, default 0
            Coefficient of the activity regularizer

        - d_pooled: list, default [3]
            Number of cells pooled in max_pooling for CellCnn_Ensemble

        - dropout_prob: float, default 0.5
            dropout rate

        - lr: float, default 1e-4
            learning rate used for the optimizers

        - num_top: int, default 1
            How many experts to use to generate a cell marker values

        - noisy_gating: bool, default True
            Whether to add noise during training to gating weights

        - noise_eps: float, default 1e-2
            noise threshold

        - beta_1: float, default 0.9
            beta1 value for Adam Optimizer

        - beta_2: float, default 0.999
            beta2 value for Adam Optimizer

        - reg_lambda: float, default 10
            gradient penalty regularizer for WGAN

        - train: bool, default False
            whether we are training or testing, adds noise to gating if True

        - init_method: str, default 'xavier'
            Method for initializing weights

        - type_gan: str, default 'Normal'
            Type of GAN used

    """

    def __init__(self,
                 noise_size,
                 moe_sizes,
                 batch_size,
                 num_markers=10,
                 num_experts=3,
                 g_filters=10,
                 d_filters=[10],
                 d_pooled=[3],
                 coeff_l1=0,
                 coeff_l2=1e-4,
                 coeff_act=0,
                 dropout_prob=0.5,
                 lr=1e-4,
                 num_top=1,
                 noisy_gating=True,
                 noise_eps=1e-2,
                 beta_1=0.9,
                 beta_2=0.999,
                 reg_lambda=10,
                 train=False,
                 init_method='xavier',
                 type_gan='normal',
                 load_balancing=False):

        # Initialize the generator and discriminator

        self.generator = CellGanGen(
            moe_sizes=moe_sizes,
            num_markers=num_markers,
            num_experts=num_experts,
            num_top=num_top,
            noisy_gating=noisy_gating,
            noise_epsilon=noise_eps,
            num_filters=g_filters,
            init_method=init_method)

        self.discriminator = CellCnnEnsemble(
            d_filters=d_filters,
            coeff_l1=coeff_l1,
            coeff_l2=coeff_l2,
            coeff_act=coeff_act,
            d_pooled=d_pooled,
            init_method=init_method,
            dropout_prob=dropout_prob)
        self.num_discriminators = len(d_filters)

        self.noise_size = noise_size
        self.batch_size = batch_size

        self.hparams = dict()
        self.hparams['type_gan'] = type_gan
        self.hparams['load_balancing'] = load_balancing
        self.hparams['reg_lambda'] = reg_lambda
        self.hparams['train'] = train

        # Optimizer Params
        if self.hparams['type_gan'] != 'wgan':
            self.adam_optimizer = dict()
            self.adam_optimizer['learning_rate'] = lr
            self.adam_optimizer['beta_1'] = beta_1
            self.adam_optimizer['beta_2'] = beta_2

        else:
            self.rms_prop_optimizer = dict()
            self.rms_prop_optimizer['learning_rate'] = lr

        self._create_placeholders()
        self._compute_loss()
        self._solvers()

    def set_train(self, train):
        """
        Toggle for whether we are in training or testing stage, controls noise addition
        to gating weights

        :param train: bool, whether in training or testing stage
        :return: no returns
        """

        self.hparams['train'] = train

    def _create_placeholders(self):
        """
        Creates the placeholders Z and X for the input
        noise and real samples

        """

        num_markers = self.generator.hparams['num_markers']

        self.Z = tf.placeholder(
            shape=[None, None, self.noise_size],
            dtype=tf.float32,
            name="input_noise")

        self.X = tf.placeholder(
            shape=[None, None, num_markers],
            dtype=tf.float32,
            name="Real_samples")

        self.shape = tf.shape(self.X)

    def _generate_sample(self, reuse=tf.AUTO_REUSE):
        """
        Generates a sample from the generator for given input noise
        :param reuse: bool/reuse object, indicating whether to reuse existing
                      generator parameters
        :return: output, a tensor of expected shape (batch_size, num_cells_per_input, num_markers)
        """

        output = self.generator.build_gen(
            inputs=self.Z, train=self.hparams['train'], reuse=reuse)

        return output

    def _eval_disc(self, inputs, reuse):
        """
        Returns the fake/real scores from the CellCnn Ensemble
        :param inputs: tensor, of shape (batch_size, num_cells_per_input, num_markers)
        :param reuse: bool/reuse object, indicating whether to reuse existing
                      generator parameters
        :return: output, dictionary of tensors with discriminator scores for every cell
        """

        output = self.discriminator._build_ensemble(inputs=inputs, reuse=reuse)
        return output

    def _calc_grad_norm(self, ys, xs):
        """
        Calculates the gradient, used in the wgan-gp loss formulation
        :param ys: input tensor (shape: batch_size, num_cells_per_input, num_markers)
        :param xs: bool/reuse object, indicating reuse of existing variables and parameters
        :return: no returns
        """
        self.grad = tf.gradients(ys=ys, xs=[xs])[0]
        self.grad_norm = tf.sqrt(tf.reduce_sum(self.grad**2, axis=1))

    def _compute_loss(self):
        """
        Computes the loss for the GAN based on the type used

        :return: no returns
        """

        self.g_sample = self._generate_sample()
        self.d_real = self._eval_disc(inputs=self.X, reuse=tf.AUTO_REUSE)
        self.d_fake = self._eval_disc(
            inputs=self.g_sample, reuse=tf.AUTO_REUSE)

        if self.hparams['type_gan'] == 'normal':

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

        elif self.hparams['type_gan'] == 'wgan':

            self.d_loss_fake = 0
            self.d_loss_real = 0

            for i in self.d_fake:
                self.d_loss_fake += tf.reduce_mean(self.d_fake[i])

            for i in self.d_real:
                self.d_loss_real += tf.reduce_mean(self.d_real[i])

            self.d_loss = self.d_loss_fake - self.d_loss_real

            if self.hparams['load_balancing']:
                self.g_loss = -self.d_loss_fake + self.generator.moe_loss
            else:
                self.g_loss = -self.d_loss_fake

        elif self.hparams['type_gan'] == 'wgan-gp':

            self.d_loss_real = 0
            self.d_loss_fake = 0

            epsilon = tf.random_uniform(
                shape=[self.batch_size], minval=0, maxval=1)
            self.x_hat = epsilon * tf.transpose(
                self.X) + (1 - epsilon) * tf.transpose(self.g_sample)
            self.x_hat = tf.transpose(self.x_hat)

            self.d_x_hat = self._eval_disc(
                inputs=self.x_hat, reuse=tf.AUTO_REUSE)

        else:
            raise NotImplementedError(
                'Loss for GAN of type {} is not implemented'.format(
                    self.hparams['type_gan']))

        self.d_params = tf.get_collection(
            key=tf.GraphKeys.GLOBAL_VARIABLES, scope="CellCnnEnsemble")

        self.g_params = tf.get_collection(
            key=tf.GraphKeys.GLOBAL_VARIABLES, scope="CellGanGen")

    def _solvers(self):
        """
        Optimizers used for minimizing GAN loss

        :return: no returns
        """

        # WGAN formulation used RMS Prop optimizer

        if self.hparams['type_gan'] == 'wgan':

            d_opt = tf.train.RMSPropOptimizer(
                learning_rate=self.rms_prop_optimizer['learning_rate'])
            g_opt = tf.train.RMSPropOptimizer(
                learning_rate=self.rms_prop_optimizer['learning_rate'])

            self.d_solver = d_opt.minimize(
                loss=self.d_loss, var_list=self.d_params)

            self.g_solver = g_opt.minimize(
                loss=self.g_loss, var_list=self.g_params)

            self.clip_D = [
                p.assign(tf.clip_by_value(p, -0.01, 0.01))
                for p in self.d_params
            ]

        # Use Adam Optimizer otherwise
        else:
            d_opt = tf.train.AdamOptimizer(
                learning_rate=self.adam_optimizer['learning_rate'],
                beta1=self.adam_optimizer['beta_1'],
                beta2=self.adam_optimizer['beta_2'])
            g_opt = tf.train.AdamOptimizer(
                learning_rate=self.adam_optimizer['learning_rate'],
                beta1=self.adam_optimizer['beta_1'],
                beta2=self.adam_optimizer['beta_2'])

            self.d_solver = d_opt.minimize(
                loss=self.d_loss, var_list=self.d_params)

            self.g_solver = g_opt.minimize(
                loss=self.g_loss, var_list=self.g_params)


# TODO: Add the formulations for KL divergence, heat maps, pre-training clustering
# TODO: Fix plotting issues
# TODO: Make clip_value a parameter as well
