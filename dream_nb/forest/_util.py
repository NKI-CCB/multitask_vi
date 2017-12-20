import numpy as np
import numba

from matplotlib import pyplot as plt

from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_array

from sklearn.metrics import mean_squared_error


def calculate_proximity(model, X):
    """Calculates proximity for X using (trained) model."""
    leaf_nodes = model.apply(X)
    return _calculate_proximity(leaf_nodes)


@numba.njit
def _calculate_proximity(leaf_nodes):
    # Create result matrix.
    num_feat, num_trees = leaf_nodes.shape
    proximity = np.identity(num_feat, np.int16) * num_trees

    # For each pair of features
    for i in range(num_feat):
        for j in range(i + 1, num_feat):
            # Calculcate number of co-occurrences.
            co_occ = np.sum(leaf_nodes[i, :] == leaf_nodes[j, :])

            # Store result.
            proximity[i, j] = co_occ
            proximity[j, i] = co_occ

    return proximity


def calculate_perm_vi(model, X, y, sample_weight=None,
                      sampling_weight=None, normalize=False):
    """Computes permutation VI score for given model, X and y."""

    #if y.ndim == 1:
    #    # reshape is necessary to preserve the data contiguity against vs
    #    # [:, np.newaxis] that does not.
    #    y = np.reshape(y, (-1, 1))

    X = check_array(X, dtype=DTYPE, accept_sparse='csr')

    n_samples = y.shape[0]
    n_features = X.shape[1]

    vi = np.zeros((len(model.estimators_), n_features), dtype=np.float32)

    for t, estimator in enumerate(model.estimators_):
        # Extract oob features and response values.
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state, n_samples, sampling_weight)

        X_unsampled = X[unsampled_indices, :]
        y_unsampled = y[unsampled_indices]

        if sample_weight is None:
            weight_unsampled = None
        else:
            weight_unsampled = sample_weight[unsampled_indices]

        # Calculate MSE.
        y_estimator = estimator.predict(X_unsampled, check_input=False)
        mse = mean_squared_error(y_unsampled, y_estimator)

        # Permute variable in X.
        for i in range(n_features):
            # Copy and shuffle feature values.
            f_orig = np.array(X_unsampled[:, i])
            np.random.shuffle(X_unsampled[:, i])

            # Calculate permuted MSE.
            y_estimator_perm = estimator.predict(X_unsampled, check_input=False)
            mse_perm = mean_squared_error(y_unsampled, y_estimator_perm,
                                          sample_weight=weight_unsampled)

            # Restore unpermuted feature values.
            X_unsampled[:, i] = f_orig

            # Store difference for feature i in tree t.
            vi[t, i] = max(0, mse_perm - mse)

    # Calculate overall VI score.
    score = np.mean(vi, axis=0)

    if normalize:
        score /= np.sum(score)

    return score


def calculate_sampling_weights(model, X, y, x0):
    """Calculates sampling weights based on proximity to x0.

    Args:
        model (ForestClassifier or ForestRegressor):
            Forest model that should be used to esimtate
            proximity.
        X (array-like or sparse matrix): The input samples.
            Shape=(n_samples, n_features).
        y (array-like): Input response/classes.
        x0 (int, array-like[int]): Index or list of indices of sample(s)
            to use as reference samples.

    Returns:
        array[float]: Array of sampling probabilities,
            based on proximity to x0. Shape = n_samples.

    """

    # Wrap single x0 in list.
    if isinstance(x0, int):
        x0 = [x0]

    # Fit model.
    model.fit(X=X, y=y)

    # Calculate proximity.
    proximity = calculate_proximity(model, X=X)

    # Calculate sampling probability.
    weights = np.mean(proximity[x0, :], axis=0)
    weights /= np.sum(weights)

    return weights


def calculate_perm_pair_vi(model, X, y, sample_weight=None,
                           sampling_weight=None):
    """Computes pairwise permutation VI score for given model, X and y."""

    #if y.ndim == 1:
    #    # reshape is necessary to preserve the data contiguity against vs
    #    # [:, np.newaxis] that does not.
    #    y = np.reshape(y, (-1, 1))

    X = check_array(X, dtype=DTYPE, accept_sparse='csr')

    n_samples = y.shape[0]
    n_features = X.shape[1]

    vi = np.zeros((len(model.estimators_), n_features, n_features), dtype=np.float32)

    for t, estimator in enumerate(model.estimators_):
        # Extract oob features and response values.
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state, n_samples, sampling_weight)

        X_unsampled = X[unsampled_indices, :]
        y_unsampled = y[unsampled_indices]

        if sample_weight is None:
            weight_unsampled = None
        else:
            weight_unsampled = sample_weight[unsampled_indices]

        # Calculate MSE.
        y_estimator = estimator.predict(X_unsampled, check_input=False)
        mse = mean_squared_error(y_unsampled, y_estimator)

        # Permute variable in X.
        for i in range(n_features):
            i_orig = np.array(X_unsampled[:, i])
            np.random.shuffle(X_unsampled[:, i])

            for j in range(i, n_features):
                # Copy and shuffle feature values.
                j_orig = np.array(X_unsampled[:, j])
                np.random.shuffle(X_unsampled[:, j])

                # Calculate permuted MSE.
                y_estimator_perm = estimator.predict(X_unsampled, check_input=False)
                mse_perm = mean_squared_error(y_unsampled, y_estimator_perm,
                                              sample_weight=weight_unsampled)

                # Restore unpermuted feature values.
                X_unsampled[:, j] = j_orig

                # Store difference for feature i in tree t.
                vi[t, i, j] = max(0, mse_perm - mse)

            X_unsampled[:, i] = i_orig

    # Calculate overall VI score.
    score = np.mean(vi, axis=0)

    return score


def calculate_cond_pair_vi(model, X, y, sample_weight=None,
                           sampling_weight=None):
    """Computes pairwise permutation VI score for given model, X and y."""

    #if y.ndim == 1:
    #    # reshape is necessary to preserve the data contiguity against vs
    #    # [:, np.newaxis] that does not.
    #    y = np.reshape(y, (-1, 1))

    X = check_array(X, dtype=DTYPE, accept_sparse='csr')

    n_samples = y.shape[0]
    n_features = X.shape[1]

    vi = np.zeros((len(model.estimators_), n_features, n_features),
                  dtype=np.float32)

    for t, estimator in enumerate(model.estimators_):
        # Extract oob features and response values.
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state, n_samples, sampling_weight)

        X_unsampled = X[unsampled_indices, :]
        y_unsampled = y[unsampled_indices]

        if sample_weight is None:
            weight_unsampled = None
        else:
            weight_unsampled = sample_weight[unsampled_indices]

        # Calculate MSE.
        y_estimator = estimator.predict(X_unsampled, check_input=False)
        mse = mean_squared_error(y_unsampled, y_estimator)

        # Copy X for second permutation.
        X_copy = np.copy(X_unsampled)

        # Permute variable in X.
        for i in range(n_features):
            i_orig = np.array(X_unsampled[:, i])
            np.random.shuffle(X_unsampled[:, i])

            # MSE of permuted i.
            y_perm_i = estimator.predict(X_unsampled, check_input=False)
            mse_i = mean_squared_error(y_unsampled, y_perm_i,
                                       sample_weight=weight_unsampled)

            for j in range(i, n_features):
                # Copy and shuffle feature values.
                j_orig = np.array(X_unsampled[:, j])
                np.random.shuffle(X_unsampled[:, j])

                X_copy[:, j] = X_unsampled[:, j]

                # MSE of permuted j.
                y_perm_j = estimator.predict(X_copy, check_input=False)
                mse_j = mean_squared_error(y_unsampled, y_perm_j,
                                           sample_weight=weight_unsampled)

                # MSE of permuted i and j.
                y_perm_both = estimator.predict(X_unsampled, check_input=False)
                mse_both = mean_squared_error(y_unsampled, y_perm_both,
                                              sample_weight=weight_unsampled)

                # Restore unpermuted feature values.
                X_unsampled[:, j] = j_orig
                X_copy[:, j] = j_orig

                # Store difference for feature i in tree t.
                cond_vi = min((mse_both - mse_i), (mse_both - mse_j))
                vi[t, i, j] = max(0, cond_vi)

            X_unsampled[:, i] = i_orig

    # Calculate overall VI score.
    score = np.mean(vi, axis=0)

    return score


def plot_sampling_weights(weights, ax=None, sort=False, **kwargs):
    if ax is None:
        _, ax = plt.subplots()

    if sort:
        weights = sorted(weights)

    ax.plot(weights, '.', **kwargs)
    ax.axhline(1 / len(weights), linestyle='dashed', color='#c44e52')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Resampling probability')

    return ax
