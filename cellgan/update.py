import pandas as pd
import os
import json
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DEFAULT_INHIBITOR = 'AKTi'
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'results')


def load_hparams_from_exp(save_dir, exp_name):
    """Loads hyperparameters for given experiment"""
    hparams_file = os.path.join(save_dir, exp_name, 'Hparams.txt')
    with open(hparams_file, 'r') as f:
        hparams = json.load(f)
    return hparams


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inhibitor', default=DEFAULT_INHIBITOR,
                        help='Inhibitor to update experiment details for')

    parser.add_argument('--out_dir', dest='output_dir', default=DEFAULT_OUT_DIR,
                        help='Base output directory')

    parser.add_argument('--method', default='cellgan', choices=['cellgan', 'baseline'],
                        help='Whether the experiment was cellgan or other baseline method')

    parser.add_argument('--baseline_method', default='gmm', help='Which baseline method used.')
    args = parser.parse_args()

    if args.method == 'baseline':
        save_dir = os.path.join(DEFAULT_OUT_DIR, 'baselines', args.baseline_method, args.inhibitor)
    else:
        save_dir = os.path.join(DEFAULT_OUT_DIR, 'cellgan', args.inhibitor)

    filename = args.inhibitor + '_experiments.xlsx'
    experiments = os.listdir(save_dir)

    sample = experiments[0]
    hparams = load_hparams_from_exp(save_dir, sample)
    hparams_all = {param: list() for param in hparams}

    for experiment in experiments:
        hparams = load_hparams_from_exp(save_dir, experiment)
        for param in hparams:
            hparams_all[param].append(hparams[param])

    exp_df = pd.DataFrame.from_dict(hparams_all)
    exp_df.to_excel(os.path.join(save_dir, filename))


if __name__ == '__main__':
    main()
