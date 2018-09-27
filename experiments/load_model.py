
import tensorflow as tf
from model import CellGan
import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------
# Loading CellGan parameters
# --------------------------


def load_model(model_dir, regex_folder):

    h_params_file = os.path.join(model_dir, 'Hparams.txt')
    with open(h_params_file, 'r') as f:
        CellGan_params = json.load(f)

    train = True

    model_name = 'model' + regex_folder + '.ckpt'
    model_path = os.path.join(model_dir, model_name)

    with tf.Session() as sess:

        print("Building CellGan...\n")

        model = CellGan(
            noise_size=CellGan_params['noise_size'],
            batch_size=CellGan_params['batch_size'],
            moe_sizes=CellGan_params['moe_sizes'][1:-1],
            num_experts=CellGan_params['num_experts'],
            ncell=CellGan_params['ncell'],
            nmark=CellGan_params['nmark'],
            gfilter=CellGan_params['gfilter'],
            dfilters=CellGan_params['dfilters'],
            lr=CellGan_params['learning_rate'],
            train=train,
            beta1=CellGan_params['beta1'],
            beta2=CellGan_params['beta2'],
            n_top=CellGan_params['n_top'],
            reg_lambda=CellGan_params['reg_lambda'],
            typeGAN=CellGan_params['typeGAN'],
            npooled=CellGan_params['npooled']
        )

        saver = tf.train.Saver()
        print("Loading Model: {} ...\n".format(model_name))
        saver.restore(sess, model_path)
        print("Model {} loaded \n".format(model_name))
        print("CellGan Built for further analysis. \n")

    return model, CellGan_params


if __name__ == '__main__':

    # Test to see if the code is running

    inhibitor_used = 'AKTi'
    regex_well = '_A05'
    test = 1

    hard_drive = os.path.join(sys.path[0], 'Real Data Tests')

    if test < 10:
        regex_folder = regex_well + '_0' + str(test)
    else:
        regex_folder = regex_well + '_' + str(test)

    model_dir = os.path.join(hard_drive, inhibitor_used, regex_folder)

    if not os.path.exists(model_dir):
        raise NotADirectoryError('Directory does not exist. Please check provided inputs')

    model, CellGan_params = load_model(model_dir=model_dir, regex_folder=regex_folder)