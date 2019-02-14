import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_INPUT_DIR = os.path.join(DATA_DIR, DEFAULT_INHIBITOR)
DEFAULT_FCS_FILE = os.path.join(DEFAULT_INPUT_DIR, DEFAULT_INHIBITOR + '_fcs.csv')
DEFAULT_MARKERS_FILE = os.path.join(DEFAULT_INPUT_DIR, 'markers.csv')
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results/cellgan', DEFAULT_INHIBITOR)
DEFAULT_MARKERS = [
    'CD3', 'CD45', 'CD4', 'CD20', 'CD33', 'CD123', 'CD14', 'IgM', 'HLA-DR',
    'CD7'
]
