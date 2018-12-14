import numpy as np
import tensorflow as tf
import sys
import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from lib.utils import compute_ks
from lib.model import CellGan
from lib.utils import sample_z
from experiments.load_model import load_model

DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/bodenmiller/cellgan', DEFAULT_INHIBITOR, '17-11_19-09-11')
DEFAULT_NUM_SAMPLES = 8350

with tf.Session() as sess:

    model, hparams = load_model(out_dir=DEFAULT_OUT_DIR, session_obj=sess)

    assert isinstance(model, CellGan)

    noise_sample = sample_z(
        batch_size=1,
        num_cells_per_input=DEFAULT_NUM_SAMPLES,
        noise_size=hparams['noise_size'])

    fetches = [model.g_sample, model.generator.gates, model.generator.logits]
    feed_dict = {model.Z: noise_sample}

    fake_samples, gates, logits = sess.run(
        fetches=fetches, feed_dict=feed_dict)
    fake_samples = fake_samples.reshape(DEFAULT_NUM_SAMPLES, hparams['num_markers'])

    fake_sample_experts = np.argmax(gates, axis=1)

    expert_expert_ks = compute_ks(real_data=fake_samples, real_labels=fake_sample_experts,
                                  fake_data=fake_samples, expert_labels=fake_sample_experts,
                                  num_subpopulations=hparams['num_experts'],
                                  num_experts=hparams['num_experts'])







