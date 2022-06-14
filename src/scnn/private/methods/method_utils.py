"""Utilities for initializing optimization methods."""

from typing import Union, Tuple, Optional, List
import math

from scipy.sparse.linalg import eigsh  # type: ignore

import lab

from scnn.private.models.model import Model


def init_batch_size(
    train_set: Tuple[lab.Tensor, lab.Tensor], batch_size: int = 128
) -> int:
    """
    :param train_set: an (X,y) tuple containing the training set.
    :param batch_size: the batch size to use. If less than 1, it is assumed that the
        provided floating point number is the fraction of the training set which the
        the mini-batch should compose.
    :returns: the desired batch size.
    """
    # only floating point or integer numbers are permitted.
    assert isinstance(batch_size, float) or isinstance(batch_size, int)

    if batch_size == -1:  # full batch
        return train_set[0].shape[0]
    elif batch_size < 1.0:
        return max(int(math.floor(train_set[0].shape[0] * batch_size)), 1)
    else:
        return batch_size


def init_max_epochs(
    train_set: Tuple[lab.Tensor, lab.Tensor],
    max_epochs: Optional[int] = None,
    max_iters: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> int:
    """Determine the maximum number of epochs to run an optimization algorithm
    given either either the number of epochs directly or the number of
    iterations desired.

    :param train_set: an (X,y) tuple containing the training set.
    :param max_epochs: the number of epochs to run. This will be return directly if it is non 'None'.
    :param max_iters: the maximum number of (inner) iterations to run the optimization algorithm.
        This can be useful to provide if the goal is to run a fixed number of iterations regardless
        of the dataset size.
    :param batch_size: the batch size that will be used by the optimizer.
        This must be provided if 'max_iters' is not None.
    """
    # only one of max_epochs or max_iters may be provided.
    assert max_epochs is None or max_iters is None

    if max_epochs is not None:
        return max_epochs
    elif batch_size is None:
        return max_iters
    else:
        return int(math.ceil(max_iters * batch_size / train_set[0].shape[0]))
