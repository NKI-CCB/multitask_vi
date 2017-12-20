import numpy as np
import pandas as pd


def inverse_onehot(data, column_sets, drop=False):
    """Performs the inverse onehot operation for groups of columns."""

    # Ensure we are working on a copy.
    data = data.copy()

    # Extract columns.
    inverse = {}
    for set_name, set_columns in column_sets.items():
        inverse[set_name] = _inverse_onehot(data[set_columns])
        data.drop(set_columns, inplace=True, axis=1)

    if drop:
        # Return separate data + inverse mapping.
        inverse_df = pd.DataFrame(inverse)
        return data, inverse_df
    else:
        # Inject inverse mapping into frame.
        for set_name, inverse_col in inverse.items():
            data[set_name] = inverse_col

        return data


def _inverse_onehot(data):
    # Ensure data is boolean.
    data = data.astype(bool)

    # Check form.
    if any(data.sum(axis=1) > 1):
        raise ValueError('Rows contain multiple True values')

    # Identify corresponding index, set to NaN if not present.
    column = data.idxmax(axis=1)
    column[~data.any(axis=1)] = np.nan

    return column
