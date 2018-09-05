
import tensorflow as tf
from utils import initializers
from expert_utils import ffn_expert_fn, local_moe


class CellGan_gen(object):

    """
    Creates an object of CellGan_gen, which is the generator class for CellGan.

    Args:
        - moe_sizes: list of ints
            Size of the hidden layer used for the MoE

        - num_experts: int
            Number of experts used in the CellGan_generator

        - ncell: int
            Number of cells per multi-cell input

        - nmark: int
            Number of markers used in the experiment

        - gfilter: int
            Number of filters to be used in convolutional layer

        - noisy_gating: bool
            Whether to use the noise component in gating networks

        - noise_epsilon: float
            noise threshold

        - n_top: int
            Number of top experts to use for each example

        - init_method: str, default 'xavier'
            Method of initializing the weights

    """

    def __init__(self, moe_sizes, num_experts, ncell, nmark, gfilter,
                 noisy_gating, noise_epsilon, n_top, init_method):

        self.experts = dict()

        self.hparams = dict()
        self.hparams['moe_sizes'] = moe_sizes
        self.hparams['num_experts'] = num_experts
        self.hparams['ncell'] = ncell
        self.hparams['nmark'] = nmark
        self.hparams['n_top'] = n_top
        self.hparams['nfilter'] = gfilter

        self.gating = dict()
        self.gating['noisy'] = noisy_gating
        self.gating['noise_eps'] = noise_epsilon

        self.init = initializers[init_method]
        self.moe_func = None
        self.moe_loss = None
        self.moe_input_size = None
        self.gates = None,
        self.load = None

    def build_gen(self, inputs, train, reuse):

        """
        Generates a multi-cell input for the given sample

        Inputs
        ------

        inputs:     tensor, (batch_size, ncell, noise_size)
                    - Input noise

        train:      bool,
                    - No noise added in MoE, if False, added otherwise

        reuse:      bool/Reuse object,
                    - Indicating whether the created variables can be
                    reused

        Returns
        -------

        moe_output: Output of the generator for given noise

        """
        
        num_experts = self.hparams['num_experts']
        shape_check = False

        with tf.variable_scope("CellGan_gen", reuse=reuse):

            batch_size = tf.shape(inputs)[0]
            ncell = tf.shape(inputs)[1]
            noise_size = int(inputs.shape[-1])

            inval = tf.reshape(
                inputs,
                shape=[batch_size * ncell, noise_size, 1]
            )

            conv1 = tf.layers.conv1d(
                inputs=inval,
                filters=self.hparams['nfilter'],
                kernel_initializer=self.init(),
                kernel_size=noise_size,
                activation=tf.nn.relu,
                name="g_conv1",
            )

            reshape = tf.reshape(
                conv1,
                shape=[batch_size*ncell,
                       self.hparams['nfilter']]
            )

            self.moe_input_size = int(reshape.shape[-1])

            self.moe_func = ffn_expert_fn(
                input_size=self.moe_input_size,
                hidden_sizes=self.hparams['moe_sizes'],
                output_size=self.hparams['nmark'],
                hidden_activation=tf.nn.leaky_relu
            )

            moe_output, self.moe_loss, self.gates, self.load = local_moe(
                x=reshape,
                train=train,
                expert_fn=self.moe_func,
                num_experts=num_experts,
                k=self.hparams['n_top'],
                pass_x=True,
                loss_coef=100,
                name="g_moe"
            )

            reshaped_moe_output = tf.reshape(
                moe_output,
                shape=[batch_size, ncell, self.hparams['nmark']]
            )

            # Checking shapes

            if shape_check:
                print("GENERATOR")
                print(inval.shape)
                print(conv1.shape)
                print(reshape.shape)
                print(moe_output)
                print(reshaped_moe_output.shape)

        return reshaped_moe_output

    def get_moe_input_size(self):

        """ Returns the number of input neurons in an expert"""

        return self.moe_input_size
