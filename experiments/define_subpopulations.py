import numpy as np
import tensorflow as tf
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from lib.utils import sample_z, compute_wasserstein, load_model, compute_ks, build_logger
from lib.plotting import plot_expert_vs_expert_markers
from lib.model import CellGan

DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/bodenmiller/cellgan', DEFAULT_INHIBITOR, '17-11_19-09-11')
DEFAULT_NUM_SAMPLES = 8350

DEFAULT_MARKERS = [
    'CD3', 'CD45', 'CD4', 'CD20', 'CD33', 'CD123', 'CD14', 'IgM', 'HLA-DR',
    'CD7'
]

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

    plot_expert_vs_expert_markers(out_dir=DEFAULT_OUT_DIR, fake_subset=fake_samples,
                                  fake_subset_labels=fake_sample_experts, num_experts=hparams['num_experts'],
                                  num_markers=len(DEFAULT_MARKERS), marker_names=DEFAULT_MARKERS, zero_sub=True)
