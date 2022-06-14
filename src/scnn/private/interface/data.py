"""Process raw data into a format suitable for optimization.

TODO:
    - decide on the format for external data and provide this somewhere, possibly as part of a public `types` module.
"""

import lab

from scnn.private.utils.data.transforms import (
    unitize_columns,
    add_bias_col,
)


def normalized_into_input_space(
    weights: lab.Tensor, column_norms: lab.Tensor
) -> lab.Tensor:
    """Map the weights of a model in the column-normalized data space into the
    original data (ie. as input) space.

    Args:
        weights: a :math:`(\\ldots \\times d)` tensor of weights to map back into the input space.
        column_norms: a :math:`(d,)` vector of the column-norms of the original training data.

    Returns:
        A :math:`(\\ldots \\times d)` tensor of weights in the original
        input space.
    """
    return weights / column_norms


def input_into_normalized_space(model_weights, column_norms):
    """Map the weights of a model in the original data space into the column-
    normalized data space.

    Args:
        weights: a :math:`(\\ldots \\times d)` tensor of weights to map into the column-normalized space.
        column_norms: a :math:`(d)` vector of the column-norms of the original training data.

    Returns:
        A :math:`(\\ldots \\times d)` tensor of weights in the column-normalized space.
    """

    return model_weights * column_norms


def process_data(
    X_train,
    y_train,
    X_test,
    y_test,
    unitize_data: bool = True,
    bias: bool = False,
):
    """Process training and test data into a format suitable for optimization.

    The data can be input as a list of lists or in any format implementing a `to_list` method
    that can be called to obtain a list of lists.

    Args:
        X_train: :math:`(n \\times d)` matrix of training examples.
        y_train: :math:`(n)` or :math:`(n \\times c)` vector of training targets.
        X_test: :math:`(m \\times d)` matrix of test examples.
        y_test: :math:`(m)` or :math:`(n \\times c)` vector of test targets.
        unitize_data: whether or not to unitize the column norms of the
            training matrix.
        bias: whether or not to augment the data with a bias column.

    Returns:
        The input data as lab.Tensor instances with the correct datatype.
        Targets will be upcast into :math:`(n \\times 1)` matrices
        if the number of targets is 1 and a vector is provided.
    """
    # convert from input format into list of lists.

    X_train, y_train = [
        lab.tensor(to_list(v), dtype=lab.get_dtype())
        for v in [X_train, y_train]
    ]

    if X_test is None or y_test is None:
        assert X_test is None and y_test is None
        # spoof test data with training set
        X_test = X_train
        y_test = y_train
    else:
        X_test, y_test = [
            lab.tensor(to_list(v), dtype=lab.get_dtype())
            for v in [X_test, y_test]
        ]

    # add extra target dimension if necessary
    if len(y_train.shape) == 1:
        y_train = lab.expand_dims(y_train, axis=1)
        y_test = lab.expand_dims(y_test, axis=1)

    train_set, test_set, col_norms = (X_train, y_train), (X_test, y_test), None

    if bias:
        train_set, test_set = add_bias_col(train_set), add_bias_col(test_set)

    if unitize_data:
        train_set, test_set, col_norms = unitize_columns(train_set, test_set)

    return train_set, test_set, col_norms


def to_list(v):
    """Cast a vector or matrix or list of lists into a list of lists.

    Args:
        v: the vector, matrix, or list of cast. If `v` is not a list,
        it must implement a `tolist` method which can be called to
        convert `v` into a list.

    Returns:
        the argument `v` as a list of lists.
    """
    if isinstance(v, list):
        return v
    else:
        assert hasattr(v, "tolist")
        return v.tolist()
