import tensorflow as tf
from lib.discriminator import CellCnn


class CellCnnEnsemble(object):
    """
       Creates an object of class CellCnnEnsemble, which is the discriminator for CellGan

       Args:

           - d_filters: list
               Number of filters for each of the CellCnns in the ensemble

           - d_pooled: list
               Number of cells pooled in the corresponding CellCnn

           - coeff_l1: float
               Coefficient of the l1 regularizer

           - coeff_l2: float
               Coefficient of the l2 regularizer

           - coeff_act: float
               Coefficient of the activity regularizer

           - dropout_prob: float
               dropout rate

           - init_method: str
               Method for initializing weights

       """

    def __init__(self, d_filters, d_pooled, coeff_l1, coeff_l2, coeff_act,
                 dropout_prob, init_method):

        # Add CellCnnEnsemble hyperparameters
        self.hparams = dict()
        self.hparams['d_filters'] = d_filters
        self.hparams['d_pooled'] = d_pooled
        self.hparams['coeff_l1'] = coeff_l1
        self.hparams['coeff_l2'] = coeff_l2
        self.hparams['coeff_act'] = coeff_act
        self.hparams['dropout_prob'] = dropout_prob

        self.inits = init_method
        self._initialize_ensemble()

    def _initialize_ensemble(self):
        """
        Initialize each CellCnn in the ensemble
        """

        self.CellCnns = dict()

        for i in range(len(self.hparams['d_filters'])):

            scope_name = "CellCnn_" + str(i + 1)

            self.CellCnns[i] = CellCnn(
                num_filters=self.hparams['d_filters'][i],
                num_pooled=self.hparams['d_pooled'][i],
                scope_name=scope_name,
                init_method=self.inits,
                coeff_l1=self.hparams['coeff_l1'],
                coeff_l2=self.hparams['coeff_l2'],
                coeff_act=self.hparams['coeff_act'],
                dropout_prob=self.hparams['dropout_prob'])

    def run(self, inputs, reuse=tf.AUTO_REUSE, print_shape=False):

        return self._ensemble(inputs=inputs, reuse=reuse, print_shape=print_shape)

    def _ensemble(self, inputs, reuse=tf.AUTO_REUSE, print_shape=False):
        """
        Setup the architecture of each CellCnn and return fake/real scores for inputs,
        when invoked the first time. Reuses existing architecture and returns scores, otherwise.
        The scores are returned as a dictionary where the index of CellCnn is the key.

        :param inputs: :param inputs: tensor of shape (batch_size, n_cells, n_markers)
        :param reuse: bool/reuse object, indicates if the existing architecture and params to be used
        :param print_shape: bool, indicates whether to print the shape for every CellCnn
        :return: fake/real scores for inputs, in a dictionary with CellCnn index as key.
        """

        self.outputs = dict()

        with tf.variable_scope("CellCnnEnsemble", reuse=reuse):

            for i in self.CellCnns:

                self.outputs[i] = self.CellCnns[i].run(
                    inputs=inputs, reuse=reuse, print_shape=print_shape)

        return self.outputs


if __name__ == '__main__':
    testEnsemble = CellCnnEnsemble(d_filters=[1, 2, 3], d_pooled=[1, 1, 1], coeff_l1=0, coeff_l2=0, coeff_act=0,
                                   init_method='xavier', dropout_prob=0.5)
    test_inputs = tf.placeholder(
        dtype=tf.float32, name='test_input', shape=[None, None, 20])
    testEnsemble.run(inputs=test_inputs, reuse=tf.AUTO_REUSE, print_shape=True)
