import tensorflow as tf
from lib.utils import initializers

DEFAULT_HIDDEN_UNITS = 256


class CellCnn(object):
    """
    Creates an object of class CellCnn.

    Args:

        - num_filters: int
            Number of filters used in the convolutional layer

        - coeff_l1: float
            Coefficient of the l1 regularizer

        - coeff_l2: float
            Coefficient of the l2 regularizer

        - coeff_act: float
            Coefficient of the activity regularizer

        - num_pooled: int
            Number of cells pooled in max_pooling

        - dropout_p: float
            dropout rate

        - init_method: str
            Method for initializing weights

        - scope_name: str
            Discriminator name

    """

    def __init__(self, num_filters, coeff_l1, coeff_l2, coeff_act, num_pooled,
                 dropout_prob, init_method, scope_name):

        self.scope_name = scope_name

        # Discriminator Hyperparameters
        self.hparams = dict()
        self.hparams['num_filters'] = num_filters
        self.hparams['coeff_l1'] = coeff_l1
        self.hparams['coeff_l2'] = coeff_l2
        self.hparams['coeff_act'] = coeff_act
        self.hparams['dropout_prob'] = dropout_prob
        self.hparams['num_pooled'] = num_pooled

        self.init = initializers[init_method]

    def build_disc(self, inputs, reuse=tf.AUTO_REUSE, print_shape=False):
        """
        Setup the discriminator architecture and return fake/real scores of each input, when
        invoked the first time. Reuses existing architecture and returns scores, otherwise.

        :param inputs: tensor of shape (batch_size, n_cells, n_markers)
        :param reuse: bool/reuse object, indicates if the existing architecture and params to be used
        :param print_shape: bool, indicates whether to print the shapes of different components & layers
        :return: fake/real scores for inputs, tensor of shape (batch_size*num_pooled, 1)
        """

        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Get batch_size and num_cells_per_input for ease of reshape later
            batch_size = tf.shape(inputs)[0]
            num_cells_per_input = tf.shape(inputs)[1]
            num_markers = int(inputs.shape[-1])

            # Input to the convolutional layer
            conv1_input = tf.reshape(
                inputs,
                shape=[batch_size * num_cells_per_input, num_markers, 1])

            # Output from Convolutional Layer 1
            # Expected shape: (batch_size*num_cells_per_input, 1, num_filters)
            d_conv1 = tf.layers.conv1d(
                inputs=conv1_input,
                filters=self.hparams['num_filters'],
                kernel_size=num_markers,
                kernel_initializer=self.init(),
                name='d_conv1',
                activation=tf.nn.relu,
            )

            num_filters = int(d_conv1.shape[-1])

            # Reshaped output for downstream layers
            # Expected Shape: (batch_size, num_filters, num_cells_per_input]
            reshaped_d_conv1 = tf.reshape(
                d_conv1, shape=[batch_size, num_filters, num_cells_per_input])

            # Setting an appropriate value of number of cells to be pooled
            if num_cells_per_input == 1:
                k = 1
            else:
                k = self.hparams['num_pooled']

            # Pooling layer (Instead of filter dimension, we pool on cell dimension)
            # Expected Shape: (batch_size, num_filters, num_pooled)
            d_pooled1, indices = tf.nn.top_k(
                input=reshaped_d_conv1, k=k, name='d_pooled1')

            num_pooled = int(d_pooled1.shape[-1])

            # Reshaping pooling layer output
            # Expected Shape: (batch_size*num_pooled, num_filters)
            reshaped_d_pooled1 = tf.reshape(
                d_pooled1, shape=[batch_size * num_pooled, num_filters])

            # Dense Layer 1
            # Expected Shape: (batch_size*num_pooled, DEFAULT_HIDDEN_UNITS)
            d_dense1 = tf.layers.dense(
                inputs=reshaped_d_pooled1,
                units=DEFAULT_HIDDEN_UNITS,
                name="d_dense1",
                activation=tf.nn.leaky_relu)

            # Dropout Layer 1
            # Expected Shape: (batch_size*num_pooled, DEFAULT_HIDDEN_UNITS)
            d_dropout1 = tf.layers.dropout(
                inputs=d_dense1,
                rate=self.hparams['dropout_prob'],
                name="d_dropout1")

            # Dense Layer 2
            # Expected Shape = (batch_size*num_pooled, 1)
            d_dense2 = tf.layers.dense(
                inputs=d_dropout1, units=1, name='d_dense2', activation=None)

            # For checking shapes
            if print_shape:
                print()
                print("Discriminator")
                print("-------------")
                print("Convolutional layer input shape: ", conv1_input.shape)
                print("Convolutional layer output shape: ", d_conv1.shape)
                print("Convolutional layer output reshaped: ",
                      reshaped_d_conv1.shape)
                print("Pooling layer output shape: ", d_pooled1.shape)
                print("Pooling layer output reshaped: ",
                      reshaped_d_pooled1.shape)
                print("Dense Layer 1 output shape: ", d_dense1.shape)
                print("Dense Layer 2 output shape: ", d_dense2.shape)
                print()

        return d_dense2


if __name__ == '__main__':

    testCellCnn = CellCnn(
        num_filters=10,
        coeff_l1=0,
        coeff_l2=0,
        coeff_act=0,
        dropout_prob=0.5,
        num_pooled=70,
        init_method='xavier',
        scope_name='testCellCnn')

    test_input = tf.placeholder(
        dtype=tf.float32, name='test_input', shape=[64, 100, 5])

    testCellCnn.build_disc(
        inputs=test_input, reuse=tf.AUTO_REUSE, print_shape=True)
