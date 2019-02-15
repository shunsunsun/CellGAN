import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Setting up defaults
DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_INHIB_STRENGTH = 'A02'
DEFAULT_MARKERS = ['CD3', 'CD45', 'CD4', 'CD20', 'CD33', 'CD123', 'CD14', 'IgM', 'HLA-DR','CD7']
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/baselines/GMM')
