"""Functions for calculating multitask RandomForest VI scores."""

import logging

import pandas as pd
from .vi import vi_score


def multitask_vi_score(model, X, y, design, n_jobs=1):
    """Calculates multitask VI score for a trained RandomForest model.

    VI scores are calculates using the vi_score function, which uses a
    permutation-based approach to calculate VI scores.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained random forest model.
    X : pd.DataFrame
        Training data used to train the model, as a 2D array of
        samples-by-features.
    y : numpy.ndarray or pd.Series
        Response vector used to train the model, as a 1D array of values.
    design : pd.DataFrame
        Boolean task design matrix (samples-by-task), specifying to which
        tasks samples belong.
    n_jobs : int
        Number of CPUs to use for the calculation.

    Returns
    -------
    pd.DataFrame
        Returns a pandas DataFrame of features-by-task, containing the VI
        scores of the different features across the various tasks.
    """

    assert all(design.index == X.index)

    vi_scores = {}
    for group, weights in design.items():
        logging.info('Calculating VI for task %s', group)

        vi_scores[group] = vi_score(
            model, X.values, y, n_jobs=n_jobs, sample_weight=weights)

    vi_scores_df = pd.DataFrame(vi_scores, index=X.columns)

    return vi_scores_df
