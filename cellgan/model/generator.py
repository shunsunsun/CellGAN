import tensorflow as tf
from cellgan.lib.utils import initializers
from cellgan.lib.expert_utils import ffn_expert_fn, local_moe


class CellGanGen(object):
    """
    Creates an object of CellGanGen, which is the generator class for CellGan.

    Args:
        - moe_sizes: list of ints
            sizes of the hidden layer used for the MoE

        - num_experts: int
            Number of experts used in the CellGan_generator

        - num_markers: int
            Number of markers used in the experiment

        - num_filters: int
            Number of filters to be used in convolutional layer

        - noisy_gating: bool
            Whether to use the noise component in gating networks

        - noise_epsilon: float
            noise threshold

        - num_top: int
            Number of top experts to use for each example

        - init_method: str, default 'xavier'
            Method of initializing the weights

    """

    def __init__(self,
                 moe_sizes,
                 num_experts,
                 num_markers,
                 num_filters,
                 noisy_gating,
                 noise_epsilon,
                 num_top,
                 moe_loss_coef=100,
                 init_method='xavier'):
        self.hparams = dict()
        self.hparams['moe_sizes'] = moe_sizes
        self.hparams['num_experts'] = num_experts
        self.hparams['num_markers'] = num_markers
        self.hparams['num_top'] = num_top
        self.hparams['num_filters'] = num_filters
        self.hparams['moe_loss_coef'] = moe_loss_coef

        self.gating = dict()
        self.gating['noisy'] = noisy_gating
        self.gating['noise_eps'] = noise_epsilon
        self.init = initializers[init_method]

    def run(self, inputs, train, reuse=tf.AUTO_REUSE, print_shape=False):
        return self._generator(inputs=inputs, train=train, reuse=reuse, print_shape=print_shape)

    def _generator(self, inputs, train, reuse=tf.AUTO_REUSE, print_shape=False):
        """
        Builds the generator architecture, when invoked the first time. Generates data from
        learnt distribution otherwise.

        :param inputs: tensor of shape (batch_size, ncell, noise_size)
        :param train: bool, indicates whether training stage or testing stage
                        If true, noise is added to gating weights.
        :param reuse: bool/reuse object, indicates if the existing architecture and params to be used
        :param print_shape: bool, indicates whether to print the shapes of different components & layers
        :return: generated data, a tensor of shape (batch_size, ncell, nmark)
        """
        with tf.variable_scope("CellGanGen", reuse=reuse):
            batch_size = tf.shape(inputs)[0]
            num_cells = tf.shape(inputs)[1]
            noise_size = int(inputs.shape[-1])

            # Expected Shape: (batch_size*num_cells, noise_size, 1)
            conv1_input = tf.reshape(inputs, shape=[batch_size * num_cells, noise_size, 1])

            # Expected Shape: (batch_size*num_cells, 1, num_filters)
            g_conv1 = tf.layers.conv1d(
                inputs=conv1_input,
                filters=self.hparams['num_filters'],
                kernel_initializer=self.init(),
                kernel_size=noise_size,
                activation=tf.nn.leaky_relu,
                name="g_conv1",
            )

            # Expected Shape: (batch_size*num_cells, num_filters)
            reshaped_conv1_output = tf.reshape(g_conv1, shape=[batch_size * num_cells, self.hparams['num_filters']])
            self.moe_input_size = int(reshaped_conv1_output.shape[-1])

            # Define the MoE input function
            self.moe_func = ffn_expert_fn(
                input_size=self.moe_input_size,
                hidden_sizes=self.hparams['moe_sizes'],
                output_size=self.hparams['num_markers'],
                hidden_activation=tf.nn.leaky_relu)

            # Run the mixture of experts module
            # moe_output expected shape: (batch_size*num_cells, num_markers)
            moe_output, self.moe_loss, self.gates, self.load, self.logits = local_moe(
                x=reshaped_conv1_output,
                train=train,
                expert_fn=self.moe_func,
                num_experts=self.hparams['num_experts'],
                k=self.hparams['num_top'],
                pass_x=True,
                loss_coef=self.hparams['moe_loss_coef'],
                noisy_gating=self.gating['noisy'],
                noise_eps=self.gating['noise_eps'],
                name="g_moe")

            # Expected shape: (batch_size, num_cells, num_markers)
            reshaped_moe_output = tf.reshape(moe_output, shape=[batch_size, num_cells, self.hparams['num_markers']])

            if print_shape:
                print()
                print("Generator")
                print("---------")
                print("Convolutional layer input shape: ", conv1_input.shape)
                print("Convolutional layer output shape: ", g_conv1.shape)
                print("Convolutional layer output reshaped: ",
                      reshaped_conv1_output.shape)
                print("Moe output shape: ", moe_output.shape)
                print("Reshaped moe output shape: ", reshaped_moe_output.shape)
                print()

        return reshaped_moe_output

    def get_moe_input_size(self):
        """ Returns the number of input neurons in an expert"""
        return self.moe_input_size


if __name__ == '__main__':

    testCellGanGen = CellGanGen(
        moe_sizes=[100, 100],
        num_experts=10,
        num_markers=5,
        num_filters=10,
        num_top=1,
        noisy_gating=True,
        noise_epsilon=0.05)

    test_inputs = tf.placeholder(
        dtype=tf.float32, name='test_input', shape=[None, None, 20])
    testCellGanGen.run(inputs=test_inputs, train=False, reuse=tf.AUTO_REUSE, print_shape=True)
