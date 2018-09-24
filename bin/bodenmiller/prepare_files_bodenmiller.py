from argparse import ArgumentParser
import json
import os
import sys
import re

DEFAULT_MARKERS = ['CD3', 'CD45', 'CD4', 'CD20', 'CD33', 'CD123', 'CD14', 'IgM', 'HLA-DR', 'CD7']
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def prepare_files():

    parser = ArgumentParser()

    parser.add_argument('--in_dir', dest='input_dir', default='./data')
    parser.add_argument('-i', '--inhibitor', dest='inhibitor_used', default='AKTi', help='Which inhibitor is used')
    parser.add_argument('-r', '--regex', dest='regex_well', default='A01', help='Strength of inhibitor used')
    parser.add_argument('-m', '--markers', dest='markers_of_interest', default=DEFAULT_MARKERS, type=list,
                        help='Which markers to be measured.')

    args = parser.parse_args()

    input_files = os.listdir(os.path.join(args.input_dir, args.inhibitor_used))
    files_to_save = list()
    p = re.compile(args.regex_well)

    for file in input_files:
        if bool(p.search(file)):
            files_to_save.append(file)

    fcs_save_file = args.inhibitor_used + '_fcs.csv'
    markers_save_file = 'markers.csv'

    with open(os.path.join(args.input_dir, args.inhibitor_used, fcs_save_file), "w") as f:
        f.write(json.dumps(files_to_save))

    with open(os.path.join(args.input_dir, args.inhibitor_used, markers_save_file), "w") as f:
        f.write(json.dumps(args.markers_of_interest))


if __name__ == '__main__':

    prepare_files()
