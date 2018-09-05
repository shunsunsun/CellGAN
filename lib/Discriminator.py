
import tensorflow as tf
from utils import initializers


class CellCnn(object):

    """
    Creates an object of class CellCnn

    Args:
        - ncell: int
            Number of cells per multi-cell input

        - nmark: int
            Number of markers used in the experiment

        - dfilter: int
            Number of filters used in the convolutional layer

        - coeff_l1: float
            Coefficient of the l1 regularizer

        - coeff_l2: float
            Coefficient of the l2 regularizer

        - coeff_act: float
            Coefficient of the activity regularizer

        - npooled: int
            Number of cells pooled in max_pooling

        - dropout: str
            Whether dropout is used

        - dropout_p: float
            dropout rate

        - init_method: str
            Method for initializing weights

        - scope_name: str
            Discriminator name

    """

    def __init__(self, ncell, nmark, dfilter, coeff_l1, coeff_l2,
                 coeff_act, npooled, dropout_p, init_method, scope_name):

        self.params = {}

        self.input_features = dict()
        self.input_features['ncell'] = ncell
        self.input_features['nmark'] = nmark
        self.scope_name = scope_name

        self.hparams = dict()
        self.hparams['nfilter'] = dfilter
        self.hparams['coeff_l1'] = coeff_l1
        self.hparams['coeff_l2'] = coeff_l2
        self.hparams['coeff_act'] = coeff_act
        self.hparams['dropout_p'] = dropout_p
        self.hparams['npooled'] = npooled

        self.init = initializers[init_method]
        self.indices = None

    def build_disc(self, inputs, reuse):

        """
        Produces the output of a CellCnn

        Inputs
        ------

        inputs:     tensor, shape:(batch_size, ncell, nmark)
                    - Multi cell inputs

        reuse:      bool
                    - Indicates whether defined variables need
                    to be reused

        Returns
        -------

        d_dense2:    tensor, shape:(batch_size, 1)
                    - Output from the CellCnn

        """

        shape_check = False

        with tf.variable_scope(self.scope_name, reuse=reuse):

            batch_size = tf.shape(inputs)[0]
            ncell = tf.shape(inputs)[1]

            conv1_input = tf.reshape(
                inputs,
                shape=[batch_size*ncell,
                       self.input_features['nmark'], 1]
            )

            d_conv1 = tf.layers.conv1d(
                inputs=conv1_input,
                filters=self.hparams['nfilter'],
                kernel_size=self.input_features['nmark'],
                kernel_initializer=self.init(),
                name='d_conv1',
                activation=tf.nn.relu,
            )

            conv_reshape = tf.reshape(
                d_conv1,
                shape=[batch_size, int(d_conv1.shape[-1]), ncell]
            )

            if ncell == 1:
                k = 1
            else:
                k = self.hparams['npooled']

            d_pooled1, indices = tf.nn.top_k(
                input=conv_reshape,
                k=k,
                name='d_pooled1')

            reshaped = tf.reshape(
                d_pooled1,
                shape=[batch_size*int(d_pooled1.shape[-1]),
                       int(d_pooled1.shape[1])]
            )

            d_dense1 = tf.layers.dense(
                inputs=reshaped,
                units=256,
                name="d_dense1",
                activation=tf.nn.leaky_relu
            )

            dropout_1 = tf.layers.dropout(
                inputs=d_dense1,
                rate=self.hparams['dropout_p'],
                name="d_dropout1"
            )

            d_dense2 = tf.layers.dense(
                inputs=dropout_1,
                units=1,
                name='d_dense2',
                activation=None
            )

            # For checking shapes

            if shape_check:
                print("Discriminator")
                print(conv1_input.shape)
                print(d_conv1.shape)
                print(conv_reshape.shape)
                print(d_pooled1.shape)
                print(reshaped.shape)
                print(d_dense1.shape)
                print(d_dense2.shape, "\n")

        return d_dense2
