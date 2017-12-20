import numba
import numpy as np

import matplotlib.pyplot as plt


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
