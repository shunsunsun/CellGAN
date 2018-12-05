
import numpy as np
import tensorflow as tf
import json
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)
from lib.model import CellGan


def load_model(out_dir):

    hparams_file = os.path.join(out_dir, 'Hparams.txt')
    model_path = os.path.join(out_dir, 'model.ckpt')

    with open(hparams_file, 'r') as f:
        hparams = json.load(f)

    with tf.Session() as sess:

        model = CellGan(
            noise_size=hparams['noise_size'],
            moe_sizes=hparams['moe_sizes'][1:-1],
            batch_size=hparams['batch_size'],
            num_markers=hparams['num_markers'],
            num_experts=hparams['num_experts'],
            g_filters=hparams['g_filters'],
            d_filters=np.array(hparams['d_filters']),
            d_pooled=np.array(hparams['d_pooled']),
            coeff_l1=hparams['coeff_l1'],
            coeff_l2=hparams['coeff_l2'],
            coeff_act=hparams['coeff_act'],
            num_top=hparams['num_top'],
            dropout_prob=hparams['dropout_prob'],
            noisy_gating=hparams['noisy_gating'],
            noise_eps=hparams['noise_eps'],
            lr=hparams['lr'],
            beta_1=hparams['beta_1'],
            beta_2=hparams['beta_2'],
            reg_lambda=hparams['reg_lambda'],
            clip_val=hparams['clip_val'],
            train = True,
            init_method=hparams['init_method'],
            type_gan=hparams['type_gan'],
            load_balancing=hparams['load_balancing']
        )

        saver = tf.train.Saver()
        print("Loading Model")
        saver.restore(sess, model_path)
        print("Model Loaded")

    return model, hparams
