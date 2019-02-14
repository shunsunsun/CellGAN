import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DEFAULT_INPUT_DIR = os.path.join(DATA_DIR, 'AKTi')
DEFAULT_FCS_FILE = os.path.join(DEFAULT_INPUT_DIR, 'AKTi_fcs.csv')
DEFAULT_MARKERS_FILE = os.path.join(DEFAULT_INPUT_DIR, 'markers.csv')
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/baselines/GMM', 'AKTi')

