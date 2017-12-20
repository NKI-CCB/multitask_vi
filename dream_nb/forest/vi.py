from concurrent.futures import ProcessPoolExecutor
import itertools

import numpy as np
import toolz

from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_array

from sklearn import utils as skl_utils, metrics as skl_metrics


# TODO: sampling weights.


def perm_vi(model, X, y, n_jobs=None, features=None, sample_weight=None):
    X = check_array(X, dtype=DTYPE, accept_sparse='csr')

    if features is None:
        features = range(X.shape[1])

    vi = np.zeros((len(model.estimators_), len(features)), dtype=np.float32)

    func = toolz.curry(_perm_vi_parallel, X=X, y=y,
                       features=features, sample_weight=sample_weight)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, result in enumerate(executor.map(func, model.estimators_)):
            vi[i, :] = result

    return np.nanmean(vi, axis=0)


def _extract_oob(estimator, x, y, sample_weight=None):
    unsampled = _generate_unsampled_indices(
        estimator.random_state, x.shape[0])

    x_oob = x[unsampled, :]
    y_oob = y[unsampled]
    w_oob = None if sample_weight is None else sample_weight[unsampled]

    return x_oob, y_oob, w_oob


def _perm_vi_parallel(estimator, X, y, features, sample_weight=None):
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
    n_samples, n_features = x_oob.shape
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


def pair_perm_vi(model, X, y, n_jobs=None, features=None, sample_weight=None):
    X = check_array(X, dtype=DTYPE, accept_sparse='csr')

    if features is None:
        n_features = X.shape[1]
        features = (range(n_features), range(n_features))

    vi = np.zeros((len(model.estimators_),
                   len(features[0]),
                   len(features[1])), dtype=np.float32)

    func = toolz.curry(_pair_perm_vi_parallel, X=X, y=y,
                       features=features, sample_weight=sample_weight)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, result in enumerate(executor.map(func, model.estimators_)):
            vi[i, :] = result

    return np.mean(vi, axis=0)


def _pair_perm_vi_parallel(estimator, X, y, features, sample_weight=None):
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


def cond_pair_perm_vi(model, X, y, n_jobs=None, features=None,
                      sample_weight=None):
    X = check_array(X, dtype=DTYPE, accept_sparse='csr')

    if features is None:
        n_features = X.shape[1]
        features = (range(n_features), range(n_features))

    vi = np.zeros((len(model.estimators_),
                   len(features[0]),
                   len(features[1])), dtype=np.float32)

    func = toolz.curry(_cond_pair_perm_vi_parallel, X=X, y=y,
                       features=features, sample_weight=sample_weight)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, result in enumerate(executor.map(func, model.estimators_)):
            vi[i, :] = result

    return np.mean(vi, axis=0)


def _cond_pair_perm_vi_parallel(estimator, X, y, features, sample_weight=None):
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
