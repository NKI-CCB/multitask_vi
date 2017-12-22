"""Functions for calculating basic RandomForest VI scores."""

import itertools

import pandas as pd
import numpy as np
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.tree._tree import DTYPE as TREE_DTYPE
from sklearn import utils as skl_utils, metrics as skl_metrics

from .util import parallel_map


def vi_score(model, X, y, n_jobs=1, features=None, sample_weight=None):
    """Calculates a permutation-based VI score for single features
       using the given (trained) RandomForest model.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained random forest model.
    X : numpy.ndarray or pd.DataFrame
        Training data used to train the model, as a 2D array of
        samples-by-features.
    y : numpy.ndarray or pd.Series
        Response vector used to train the model, as a 1D array of values.
    n_jobs : int
        Number of CPUs to use for the calculation.
    features : List[int]
        Optional subset of features to use, indicated by their indices in X.
    sample_weight : numpy.ndarray
        Weights to use for the different samples.

    Returns
    -------
    numpy.ndarray or pd.Series
        Array of VI scores for the various features. If X is a DataFrame,
        a pandas Series with feature annotations is returned instead of an
        array.

    """

    if isinstance(y, pd.Series):
        y = y.values

    if features is None:
        features = range(X.shape[1])

    X_arr = skl_utils.check_array(X, dtype=TREE_DTYPE, accept_sparse='csr')

    # Calculate VI scores.
    results = parallel_map(
        _vi_est,
        model.estimators_,
        X=X_arr,
        y=y,
        features=features,
        sample_weight=sample_weight,
        n_jobs=n_jobs)

    # Collect results.
    vi_scores = np.zeros(
        (len(model.estimators_), len(features)), dtype=np.float32)

    for i, result in enumerate(results):
        vi_scores[i, :] = result

    vi_scores = np.nanmean(vi_scores, axis=0)

    # Add feature names in pd.Series if df given.
    if isinstance(X, pd.DataFrame):
        vi_scores = pd.Series(vi_scores, index=X.columns)

    return vi_scores


def _vi_est(estimator, X, y, features=None, sample_weight=None):
    """Calculates permutation VI for single features using the given estimator.

    Parameters
    ----------
    estimator : sklearn.tree.tree.DecisionTreeRegressor
        Trained estimator from the RandomForest model.
    X : numpy.ndarray
        Training data used to train the model, as a 2D array of
        samples-by-features.
    y : numpy.ndarray
        Response vector used to train the model, as a 1D array of values.
    features : List[int]
        Optional subset of features to use, indicated by their indices in X.
    sample_weight : numpy.ndarray
        Weights to use for the different samples.

    Returns
    -------
    numpy.ndarray
        Array containing VI scores for the given features, based on the given
        estimator.

    """

    if features is None:
        features = range(X.shape[1])

    seed = estimator.random_state

    # Get oob features/response.
    x_oob, y_oob, sample_weight_oob = _extract_oob(
        estimator, X, y, sample_weight=sample_weight)
    x_oob_perm = skl_utils.shuffle(x_oob, random_state=seed)

    # Check if we have any oob samples.
    if np.sum(sample_weight_oob) == 0:
        return np.zeros(len(features), dtype=np.float32) * np.nan

    # Calculate MSE.
    y_pred = estimator.predict(x_oob, check_input=False)
    mse = skl_metrics.mean_squared_error(
        y_oob, y_pred, sample_weight=sample_weight_oob)

    # Allocate arrays.
    n_samples, _ = x_oob.shape
    x_tmp = np.empty(n_samples, dtype=np.float32)

    vi = np.zeros(len(features), dtype=np.float32)

    for vi_idx, i in enumerate(features):
        # Copy and shuffle feature values.
        np.copyto(x_tmp, x_oob[:, i])
        x_oob[:, i] = x_oob_perm[:, i]

        # Calculate permuted MSE.
        y_pred_perm = estimator.predict(x_oob, check_input=False)

        mse_perm = skl_metrics.mean_squared_error(
            y_oob, y_pred_perm, sample_weight=sample_weight_oob)

        vi[vi_idx] = max(0, mse_perm - mse)

        # Restore unpermuted feature values.
        x_oob[:, i] = x_tmp

    return vi


def _extract_oob(estimator, x, y, sample_weight=None):
    """Returns OOB sample data for the given estimator."""

    unsampled = _generate_unsampled_indices(estimator.random_state, x.shape[0])

    x_oob = x[unsampled, :]
    y_oob = y[unsampled]
    w_oob = None if sample_weight is None else sample_weight[unsampled]

    return x_oob, y_oob, w_oob


def pair_vi_score(model, X, y, n_jobs=1, features=None, sample_weight=None):
    """Calculates a permutation-based VI score for pairs of features
       using the given (trained) RandomForest model.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained random forest model.
    X : numpy.ndarray or pd.DataFrame
        Training data used to train the model, as a 2D array of
        samples-by-features.
    y : numpy.ndarray or pd.Series
        Response vector used to train the model, as a 1D array of values.
    n_jobs : int
        Number of CPUs to use for the calculation.
    features : List[int]
        Optional subset of features to use, indicated by their indices in X.
    sample_weight : numpy.ndarray
        Weights to use for the different samples.

    Returns
    -------
    numpy.ndarray or pd.Series
        2D array of VI scores for the feature pairs. Note that the matrix is
        symmetric and only the upper-right triangle of the matrix is filled in.
        If X is a DataFrame, a pandas DataFrame with feature annotations is
        returned instead of an array.
    """

    if isinstance(y, pd.Series):
        y = y.values

    if features is None:
        n_features = X.shape[1]
        features = (range(n_features), range(n_features))

    # Check inputs.
    X_arr = skl_utils.check_array(X, dtype=TREE_DTYPE, accept_sparse='csr')

    # Calculate VI scores.
    results = parallel_map(
        _pair_vi_est,
        model.estimators_,
        X=X_arr,
        y=y,
        features=features,
        sample_weight=sample_weight,
        n_jobs=n_jobs)

    # Collect results.
    vi_scores = np.zeros(
        (len(model.estimators_), len(features[0]), len(features[1])),
        dtype=np.float32)

    for i, result in enumerate(results):
        vi_scores[i, :] = result

    vi_scores = np.nanmean(vi_scores, axis=0)

    # Add feature names in pd.DataFrame if X is a DF.
    if isinstance(X, pd.DataFrame):
        vi_scores = pd.DataFrame(vi_scores, index=X.columns, columns=X.columns)

    return vi_scores


def _pair_vi_est(estimator, X, y, features, sample_weight=None):
    """Calculates permutation VI for feature pairs using the given estimator."""
    seed = estimator.random_state

    # Get oob features/response.
    x_oob, y_oob, sample_weight_oob = _extract_oob(
        estimator, X, y, sample_weight=sample_weight)
    x_oob_perm = skl_utils.shuffle(x_oob, random_state=seed)

    # Calculate MSE.
    y_pred = estimator.predict(x_oob, check_input=False)
    mse = skl_metrics.mean_squared_error(
        y_oob, y_pred, sample_weight=sample_weight_oob)

    # Allocate arrays.
    n_samples, _ = x_oob.shape
    i_tmp = np.empty(n_samples, dtype=np.float32)
    j_tmp = np.empty(n_samples, dtype=np.float32)

    feat_a, feat_b = features
    vi = np.zeros((len(feat_a), len(feat_b)), dtype=np.float32)

    for idx_i, i in enumerate(feat_a):
        # Permute i.
        np.copyto(i_tmp, x_oob[:, i])
        x_oob[:, i] = x_oob_perm[:, i]

        # Decide range to iterate (uses triu if ranges are the same).
        if feat_a == feat_b:
            range_b = itertools.islice(enumerate(feat_a), idx_i + 1, None)
        else:
            range_b = enumerate(feat_b)

        for idx_j, j in range_b:
            # Permute j.
            np.copyto(j_tmp, x_oob[:, j])
            x_oob[:, j] = x_oob_perm[:, j]

            # Calculate permuted mse.
            y_pred_perm = estimator.predict(x_oob, check_input=False)
            mse_perm = skl_metrics.mean_squared_error(
                y_oob, y_pred_perm, sample_weight=sample_weight_oob)

            # Calculate vi.
            vi[idx_i, idx_j] = max(0, mse_perm - mse)

            # Restore j.
            x_oob[:, j] = j_tmp

        # Restore i.
        x_oob[:, i] = i_tmp

    return vi


def conditional_pair_vi_score(model,
                              X,
                              y,
                              n_jobs=1,
                              features=None,
                              sample_weight=None):
    """Calculates a conditional, permutation-based VI score for pairs of
       features using the given (trained) RandomForest model.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained random forest model.
    X : numpy.ndarray or pd.DataFrame
        Training data used to train the model, as a 2D array of
        samples-by-features.
    y : numpy.ndarray or pd.Series
        Response vector used to train the model, as a 1D array of values.
    n_jobs : int
        Number of CPUs to use for the calculation.
    features : List[int]
        Optional subset of features to use, indicated by their indices in X.
    sample_weight : numpy.ndarray
        Weights to use for the different samples.

    Returns
    -------
    numpy.ndarray or pd.Series
        2D array of VI scores for the feature pairs. Note that the matrix is
        symmetric and only the upper-right triangle of the matrix is filled in.
        If X is a DataFrame, a pandas DataFrame with feature annotations is
        returned instead of an array.
    """

    if isinstance(y, pd.Series):
        y = y.values

    if features is None:
        n_features = X.shape[1]
        features = (range(n_features), range(n_features))

    # Check inputs.
    X_arr = skl_utils.check_array(X, dtype=TREE_DTYPE, accept_sparse='csr')

    # Calculate VI scores.
    results = parallel_map(
        _cond_pair_vi_est,
        model.estimators_,
        X=X_arr,
        y=y,
        features=features,
        sample_weight=sample_weight,
        n_jobs=n_jobs)

    # Collect results.
    vi_scores = np.zeros(
        (len(model.estimators_), len(features[0]), len(features[1])),
        dtype=np.float32)

    for i, result in enumerate(results):
        vi_scores[i, :] = result

    vi_scores = np.nanmean(vi_scores, axis=0)

    # Add feature names in pd.DataFrame if X is a DF.
    if isinstance(X, pd.DataFrame):
        vi_scores = pd.DataFrame(vi_scores, index=X.columns, columns=X.columns)

    return vi_scores


def _cond_pair_vi_est(estimator, X, y, features, sample_weight=None):
    seed = estimator.random_state

    # Get oob features/response.
    x_oob, y_oob, sample_weight_oob = _extract_oob(
        estimator, X, y, sample_weight=sample_weight)
    x_oob_perm = skl_utils.shuffle(x_oob, random_state=seed)

    # Allocate tmp/output arrays.
    n_samples, _ = x_oob.shape
    i_tmp = np.empty(n_samples, dtype=np.float32)
    j_tmp = np.empty(n_samples, dtype=np.float32)

    feat_a, feat_b = features
    vi = np.zeros((len(feat_a), len(feat_b)), dtype=np.float32)

    x_oob_copy = np.array(x_oob, copy=True)

    # Permute variable in X.
    for idx_i, i in enumerate(feat_a):
        # Permute i.
        np.copyto(i_tmp, x_oob[:, i])
        x_oob[:, i] = x_oob_perm[:, i]

        y_perm_i = estimator.predict(x_oob, check_input=False)
        mse_perm_i = skl_metrics.mean_squared_error(
            y_oob, y_perm_i, sample_weight=sample_weight_oob)

        # Decide range to iterate (uses triu if ranges are the same).
        if feat_a == feat_b:
            range_b = itertools.islice(enumerate(feat_a), idx_i + 1, None)
        else:
            range_b = enumerate(feat_b)

        for idx_j, j in range_b:
            # Permute j.
            np.copyto(j_tmp, x_oob_copy[:, j])
            x_oob_copy[:, j] = x_oob_perm[:, j]

            y_perm_j = estimator.predict(x_oob_copy, check_input=False)
            mse_perm_j = skl_metrics.mean_squared_error(
                y_oob, y_perm_j, sample_weight=sample_weight_oob)

            # Permute both.
            x_oob[:, j] = x_oob_perm[:, j]

            y_perm_ij = estimator.predict(x_oob, check_input=False)
            mse_perm_ij = skl_metrics.mean_squared_error(
                y_oob, y_perm_ij, sample_weight=sample_weight_oob)

            # Calculate vi.
            cond_vi = min((mse_perm_ij - mse_perm_i),
                          (mse_perm_ij - mse_perm_j))
            vi[idx_i, idx_j] = max(0, cond_vi)

            # Restore j.
            x_oob[:, j] = j_tmp
            x_oob_copy[:, j] = j_tmp

        # Restore i.
        x_oob[:, i] = i_tmp

    return vi
