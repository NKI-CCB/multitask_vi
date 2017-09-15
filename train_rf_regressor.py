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

    # Perform some sanity checks.
    if not all(x_train.index == y_train.index):
        raise ValueError('y_train does not match given x_train')

    # Train model.
    logging.info('Training model')

    model = RandomForestRegressor(n_jobs=args.threads,
                                  n_estimators=args.n_estimators,
                                  random_state=args.seed)
    model.fit(x_train, y_train.values.ravel())

    # Write model if requested.
    if args.output_model is not None:
        logging.info('Saving model')
        with open(args.output_model, 'wb') as file_:
            pickle.dump(model, file=file_)

    # Predict for test set (if given).
    if args.x_test is not None:
        logging.info('Reading test dataset')
        x_test = pd.read_csv(args.x_test, index_col=0)

        # Perform some sanity checks.
        #if not all(x_train.index == x_test.index):
        #    raise ValueError('x_test index does not match x_train')

        if not all(x_train.columns == x_test.columns):
            raise ValueError('x_test columns do not match x_train')

        logging.info('Calculating predictions')
        y_pred = model.predict(x_test)

        if args.output_pred is not None:
            logging.info('Writing predictions')
            y_pred_df = pd.DataFrame({'y': y_pred}, index=x_test.index)
            y_pred_df.to_csv(args.output_pred, index=True)

    logging.info('Done!')


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_train', required=True)
    parser.add_argument('--y_train', required=True)

    parser.add_argument('--x_test', default=None)
    parser.add_argument('--output_pred', default=None)

    parser.add_argument('--output_model', default=None)

    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    main()