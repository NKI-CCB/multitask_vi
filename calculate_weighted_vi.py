#!/usr/bin/env python

"""Script for calculating a combi-weighted VI matrix for given dataset(s)."""

from os import path
import sys

import argparse
import itertools
from collections import defaultdict
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dream_nb.forest.vi import perm_vi, cond_pair_perm_vi

logging.basicConfig(format='[%(asctime)-15s] %(message)s', level=logging.INFO)


def main(args):
    """Main function for generating weighted VIs for a given dataset."""

    # Load datasets.
    logging.info('Reading input datasets')
	
    x_frames = (pd.read_csv(x, index_col=0) for x in args.x_frames)
    x = pd.concat(x_frames, axis=0)

    y_frames = (pd.read_csv(y, index_col=0) for y in args.y_frames)
    y = pd.concat(y_frames, axis=0)['x']

    # Train model.
    logging.info('Training model')

    model = RandomForestRegressor(n_jobs=args.threads,
                                  n_estimators=args.n_estimators,
                                  random_state=args.seed)
    model.fit(x, y.values)

    if args.output_model is not None:
        logging.info('Saving model')
        with args.output_model.open('wb') as file_:
            pickle.dump(model, file=file_)

    # Calculate weighted VI and write as CSV.
    combi_vi = calculate_combi_weighted_vi(model, x, y, threads=args.threads)
    combi_vi.to_csv(str(args.output))

    logging.info('Done!')


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('-x', '--x_frames', nargs='+', required=True, type=Path)
    parser.add_argument('-y', '--y_frames', nargs='+', required=True, type=Path)
    parser.add_argument('-o', '--output', required=True, type=Path)
    parser.add_argument('--output_model', required=False,
                        type=Path, default=None)

    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()


def calculate_combi_weighted_vi(model, x, y, threads=1):
    """Calculates combi-weighted VIs for all combi vs non-combi features."""

    # Distinguish combi vs other (non-combi) features.
    feature_groups = group_features(x)

    combi_features = feature_groups['combi']
    other_features = list(itertools.chain.from_iterable(
        (v for k, v in feature_groups.items() if k != 'combi')))

    # Determine index of other features (needed for sklearn, which
    # accesses the data as a matrix, not as a dataframe).
    other_features_idx = [x.columns.get_loc(f) for f in other_features]

    # Calculate VIs.
    logging.info('Calculating weighted vi\'s')

    vi_scores = {}
    for combi in combi_features:
        logging.info(' - Calculating {}'.format(combi))

        weights = x[combi].values

        vi_scores[combi] = perm_vi(
            model, x.values, y.values, n_jobs=threads,
            features=other_features_idx, sample_weight=weights)

    # Aggregate in single frame.
    vi_scores_df = pd.DataFrame(vi_scores, index=other_features)

    return vi_scores_df


def group_features(x):
    """Groups features by their suffix (e.g. combi, mut, cna)."""

    features = defaultdict(list)

    for c in x.columns:
        key = c.split('_')[-1]
        features[key].append(c)

    return features


if __name__ == '__main__':
    main(parse_args())
