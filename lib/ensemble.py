import tensorflow as tf
from Discriminator import CellCnn


class CellCnn_Ensemble(object):

    """
       Creates an object of class CellCnn_Ensemble

       Args:
           - ncell: int
               Number of cells per multi-cell input

           - nmark: int
               Number of markers used in the experiment

           - nfilters: list
               Number of filters in the corresponding CellCnn

           - npooled: list
               Number of cells pooled in the corresponding CellCnn

           - coeff_l1: float
               Coefficient of the l1 regularizer

           - coeff_l2: float
               Coefficient of the l2 regularizer

           - coeff_act: float
               Coefficient of the activity regularizer

           - dropout: str
               Whether dropout is used

           - dropout_p: float
               dropout rate

           - init_method: str
               Method for initializing weights

       """

    def __init__(self, ncell, nmark, nfilters, npooled, coeff_l1,
                 coeff_l2, coeff_act, dropout_p, init_method):

        self.input_features = dict()
        self.input_features['ncell'] = ncell
        self.input_features['nmark'] = nmark

        self.hparams = dict()
        self.hparams['nfilters'] = nfilters
        self.hparams['npooled'] = npooled
        self.hparams['coeff_l1'] = coeff_l1
        self.hparams['coeff_l2'] = coeff_l2
        self.hparams['coeff_act'] = coeff_act
        self.hparams['dropout_p'] = dropout_p

        self.inits = init_method
        self._setup_ensemble()

    def _setup_ensemble(self):

        """ Setup of the ensemble of CellCnns """

        self.CellCnns = dict()

        for i in range(len(self.hparams['nfilters'])):

            scope_name = "CellCnn_" + str(i+1)

            self.CellCnns[i] = CellCnn(
                ncell=self.input_features['ncell'],
                nmark=self.input_features['nmark'],
                dfilter=self.hparams['nfilters'][i],
                npooled=self.hparams['npooled'][i],
                scope_name=scope_name,
                init_method=self.inits,
                coeff_l1=self.hparams['coeff_l1'],
                coeff_l2=self.hparams['coeff_l2'],
                coeff_act=self.hparams['coeff_act'],
                dropout_p=self.hparams['dropout_p']
            )

    def _eval_ensemble(self, inputs, reuse):

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

        self.outputs = dict()

        with tf.variable_scope("Ensemble", reuse=reuse):

            for i in self.CellCnns:

                self.outputs[i] = self.CellCnns[i].build_disc(
                    inputs=inputs,
                    reuse=reuse)

        return self.outputs
