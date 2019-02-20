import numpy as np
import os
import sys
import json
import argparse
import pandas as pd
from collections import Counter

from cellgan.lib.utils import compute_f_measure_uniformly_weighted
from cellgan.experiments.baselines.FlowSOM.defaults import *

parser = argparse.ArgumentParser()
parser.add_argument("--inhibitor", default="AKTi", help="Inhibitor used.")
parser.add_argument("--strength", default="A02", help="Strength of inhibitor used.")
args = parser.parse_args()

RESULT_DIR = os.path.join("./results/baselines/FlowSOM", args.inhibitor)
cluster_file = os.path.join(RESULT_DIR, "FlowSOM_clusters_" + args.strength + ".csv")
pred_vals = pd.read_csv(cluster_file)
clusters = np.array(pred_vals)

labels_file = os.path.join(RESULT_DIR, "Act_labels_" + args.strength + ".csv")
act_vals = pd.read_csv(labels_file)
training_labels_loaded = np.array(act_vals).reshape(-1)

print(compute_f_measure_uniformly_weighted(training_labels_loaded, clusters))
