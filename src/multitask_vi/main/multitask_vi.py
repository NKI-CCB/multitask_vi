#!/usr/bin/env python
"""Script for calculating a combi-weighted VI matrix for given dataset(s)."""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from .. import multitask_vi_score

logging.basicConfig(format='[%(asctime)-15s] %(message)s', level=logging.INFO)


def main():
    """Main function for generating weighted VIs for a given dataset."""

    args = parse_args()

    # Load datasets.
    logging.info('Reading input datasets')

    x_frames = (pd.read_csv(x, index_col=0) for x in args.x_train)
    x_train = pd.concat(x_frames, axis=0)

    y_frames = (pd.read_csv(y, index_col=0) for y in args.y_train)
    y_train = pd.concat(y_frames, axis=0)

    y_train = y_train[y_train.columns[0]]

    # Read design.
    design = pd.read_csv(args.design, index_col=0)

    # Load model.
    logging.info('Loading model')

    with args.model.open('rb') as file_:
        model = pickle.load(file_)

    # Calculate weighted VI and write as CSV.
    vi_scores = multitask_vi_score(
        model, X=x_train, y=y_train, design=design, n_jobs=args.cpus)

    logging.info('Writing output')
    vi_scores.to_csv(str(args.output))

    logging.info('Done!')


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(
        description="Calculates multitask VI score for a pre-trained model.")

    parser.add_argument(
        '--x_train',
        nargs='+',
        required=True,
        type=Path,
        help='Training data (features) in CSV format.')

    parser.add_argument(
        '--y_train',
        nargs='+',
        required=True,
        type=Path,
        help='Training data (response) in CSV format.')

    parser.add_argument(
        '--model',
        required=True,
        type=Path,
        help='Pre-trained RandomForest model.')

    parser.add_argument(
        '--design',
        required=True,
        type=Path,
        help='Design matrix describing sample/task membership (CSV format).')

    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='Output path for VI scores.')

    parser.add_argument(
        '--cpus',
        type=int,
        default=1,
        help='Number of CPUs to use for calculation.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
