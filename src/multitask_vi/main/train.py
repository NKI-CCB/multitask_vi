#!/usr/bin/env python
"""
Script for training a basic RandomForest Regressor using sklearn.

Author: Julian de Ruiter

"""

import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(format='[%(asctime)-15s] %(message)s', level=logging.INFO)


def main():
    args = parse_args()

    # Load datasets.
    logging.info('Reading train dataset')

    x_train = pd.read_csv(args.x_train, index_col=0)
    y_train = pd.read_csv(args.y_train, index_col=0)

    y_train = y_train[y_train.columns[0]]

    # Perform some sanity checks.
    if not all(x_train.index == y_train.index):
        raise ValueError('y_train does not match given x_train')

    # Train model.
    logging.info('Training model')

    model = RandomForestRegressor(
        n_jobs=args.cpus,
        n_estimators=args.n_estimators,
        random_state=args.seed)

    model.fit(x_train, y_train.values)

    # Write model if requested.
    logging.info('Saving model')

    model_path = args.output_base + '.model.pkl'
    with open(model_path, 'wb') as file_:
        pickle.dump(model, file=file_)

    # Predict for test set (if given).
    if args.x_test is not None:
        logging.info('Reading test dataset')
        x_test = pd.read_csv(args.x_test, index_col=0)

        if not all(x_train.columns == x_test.columns):
            raise ValueError('x_test columns do not match x_train')

        logging.info('Calculating predictions')
        y_pred = model.predict(x_test)

        logging.info('Writing predictions')

        pred_path = args.output_base + 'pred.csv'
        y_pred_df = pd.DataFrame({'y': y_pred}, index=x_test.index)
        y_pred_df.to_csv(pred_path, index=True)

    logging.info('Done!')


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(
        description="Trains a RandomForest regressor for the given dataset.")

    parser.add_argument(
        '--x_train',
        required=True,
        help='Training data (features) in CSV format.')
    parser.add_argument(
        '--y_train',
        required=True,
        help='Training data (response) in CSV format.')

    parser.add_argument(
        '--x_test', default=None, help='Test data (features) in CSV format.')

    parser.add_argument(
        '--output_base', required=True, help='Base output path.')

    parser.add_argument(
        '--n_estimators',
        type=int,
        default=500,
        help='Number of estimators to use when training the model.')

    parser.add_argument(
        '--cpus',
        type=int,
        default=1,
        help='Number of CPUs to use when training the model.')

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Seed to use for the RandomForest.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
