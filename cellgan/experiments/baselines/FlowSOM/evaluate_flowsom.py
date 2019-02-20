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
parser.add_argument("--nruns", default=10, type=int, help="Number of runs")
args = parser.parse_args()

RESULT_DIR = os.path.join("./results/baselines/FlowSOM", args.inhibitor, args.strength)

f_measure = list()

for run in range(1, args.nruns + 1):
    cluster_file = os.path.join(RESULT_DIR, "FlowSOM_clusters_run_" + str(run) + ".csv")
    pred_vals = pd.read_csv(cluster_file)
    clusters = np.array(pred_vals)

    labels_file = os.path.join(RESULT_DIR, "Act_labels_run_" + str(run) + ".csv")
    act_vals = pd.read_csv(labels_file)
    training_labels_loaded = np.array(act_vals).reshape(-1)

    f_measure.append(compute_f_measure_uniformly_weighted(training_labels_loaded, clusters))

print("Mean: ", np.mean(f_measure))
print("Std: ", np.std(f_measure))
