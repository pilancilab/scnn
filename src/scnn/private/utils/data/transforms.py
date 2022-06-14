"""Data transformations and related utilities.

Public Functions:

    unitize_columns: unitize the columns of a training set, optionally, a test set.


TODO:
    - Cleanup comments in this file.
"""

from typing import Tuple, Optional

import lab
import numpy as np


Dataset = Tuple[lab.Tensor, lab.Tensor]


def add_bias_col(dataset: Dataset) -> Dataset:
    """Augment the feature matrix in a dataset with a bias column.

    Args:
        dataset: a (X, y) tuple defining a dataset.
    """

    X, y = dataset

    X = lab.concatenate([X, lab.ones((X.shape[0], 1))], axis=1)

    return (X, y)


def unitize_columns(
    train_set: Dataset,
    test_set: Optional[Dataset] = None,
) -> Tuple[Dataset, Dataset, lab.Tensor]:
    """Transform a dataset so that the columns of the design matrix have unit
    norm,

    .. math::

        \\text{diag} (\\tilde X^\\top \\tilde X) = I

    If a test set is also provided, the column-norms of the training set are used to apply the same transformation to the test data.

    Args:
        train_set: an (X, y) tuple.
        test_set: (optional) an (X_test, y_test) tuple.

    Returns:
       (train_set, test_set, column_norms) --- a tuple containing the transformed training set, test set, and the column norms of the training design matrix.
       If a test_set is not provided, then `test_set` is None.
    """

    X_train = train_set[0]
    column_norms = lab.sqrt(lab.sum(X_train ** 2, axis=0, keepdims=True))

    X_train = X_train / column_norms
    train_set = (X_train, train_set[1])

    if test_set is not None:
        test_set = (test_set[0] / column_norms, test_set[1])

    return train_set, test_set, column_norms


def train_test_split(X, y, valid_prop=0.2, split_seed=1995):
    """ """
    n = y.shape[0]
    split_rng = np.random.default_rng(seed=split_seed)
    num_test = int(np.floor(n * valid_prop))
    indices = np.arange(n)
    split_rng.shuffle(indices)
    test_indices = indices[:num_test].tolist()
    train_indices = indices[num_test:].tolist()

    # subset the dataset
    X_train = X[train_indices, :]
    y_train = y[train_indices]

    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return (X_train, y_train), (X_test, y_test)
