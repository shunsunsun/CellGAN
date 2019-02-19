import numpy as np
import os
import sys
import json
import argparse
import pandas as pd

from cellgan.lib.utils import compute_f_measure_uniformly_weighted
from cellgan.lib.data_utils import load_fcs, get_fcs_filenames
from cellgan.experiments.baselines.FlowSOM.defaults import *

parser = argparse.ArgumentParser()

parser.add_argument("--in_dir", dest="input_dir", default="./data/", help="Data directory")
parser.add_argument("--inhibitor", default="AKTi", help="Inhibitor used.")
parser.add_argument("--strength", default="A02", help="Strength of inhibitor used.")
parser.add_argument("--cofactor", default=5)
parser.add_argument("--sub_limit", dest="subpopulation_limit", default=30)
args = parser.parse_args()

read_vals = pd.read_csv("./cellgan/experiments/baselines/FlowSOM/FlowSOM_clusters_" + args.strength + ".csv")
clusters = np.array(read_vals)

markers_of_interest = DEFAULT_MARKERS
fcs_files_of_interest = get_fcs_filenames(args.input_dir, args.inhibitor, args.strength)
training_data, training_labels = load_fcs(fcs_files_of_interest, markers_of_interest, args)

print(compute_f_measure_uniformly_weighted(training_labels, clusters))
