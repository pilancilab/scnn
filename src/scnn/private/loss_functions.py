"""Loss functions and related utilities.

TODO:
    - Rewrite docstrings to be Google style and correct math format.
"""
from typing import Dict, Any, List, Tuple, Union, Optional

import lab

# loss functions


def squared_error(preds: lab.Tensor, y: lab.Tensor) -> float:
    """Compute squared loss,

        $(1/2)*||preds - y||_2^2$
    :param preds: predictions.
    :param y: targets.
    returns: squared-loss
    """
    residuals = y - preds
    return lab.sum((residuals) ** 2)


def logistic_loss(preds: lab.Tensor, y: lab.Tensor):
    """Compute the logistic loss,

        $log(1 + exp(-y*preds))$
    :param preds: predictions.
    :param y: targets. These should be in {-1, 1}
    :returns: the logistic loss
    """
    res = lab.logsumexp(lab.stack([lab.zeros_like(preds), -y * preds]), axis=0)
    return lab.sum(res)


def binned_accuracy(
    preds: lab.Tensor, y: lab.Tensor, n_classes: Optional[int] = 10
):
    """Compute the accuracy of classification method by binning the
    predictions.

    :param preds: the raw predictions from the model.
    :param y: the true targets/labels.
    :param n_classes: the number of target classes. It is assumed that the classes {0, 1, ..., n_classes}.
    :returns: accuracy score.
    """

    class_preds = lab.digitize(preds, lab.arange(n_classes) + 0.5)
    return lab.sum(class_preds == y) / y.shape[0]


def accuracy(preds: lab.Tensor, y: lab.Tensor):
    """Compute the accuracy of classification, assuming y is a one-hot encoded
    vector.

    :param preds: the raw predictions from the model.
    :param y: the one-hot encoded true targets/labels
    :returns: accuracy score
    """

    class_labels = lab.argmax(y, 1)
    class_preds = lab.argmax(preds, 1)
    return lab.sum(class_preds == class_labels) / y.shape[0]


# penalty functions


def l1_penalty(w: lab.Tensor, lam: float) -> float:
    """"""
    return lam * lab.sum(lab.abs(w))


def l2_penalty(w: lab.Tensor, lam: float) -> float:
    """"""

    return (lam / 2) * lab.sum(w ** 2)


def group_l1_penalty(w: lab.Tensor, lam: Union[lab.Tensor, float]) -> float:
    """Compute the penalty associated with the regularizer.

    :param w: the parameter at which to compute group l1 penalty. Note that 'w'
        must have shape (x, P), where 'P' is the number of groups.
    :param lam: the coefficient(s) controlling the strength of regularization.
        IF 'lam' is a numpy array, it must have shape (P,).
    :returns: penalty value
    """

    return lam * lab.sum(lab.sqrt(lab.sum(w ** 2, axis=-1)))


def l1_squared_penalty(w: lab.Tensor, lam: Union[lab.Tensor, float]) -> float:
    """Compute the penalty associated with the regularizer.

    :param w: the parameter at which to compute group l1 penalty. Note that 'w'
        must have shape (x, P), where 'P' is the number of groups.
    :param lam: the coefficient(s) controlling the strength of regularization.
        IF 'lam' is a numpy array, it must have shape (P,).
    :returns: penalty value
    """

    return (lam / 2) * lab.sum(lab.sum(lab.abs(w), axis=0) ** 2)


# "activation" functions


def relu(x: lab.Tensor) -> lab.Tensor:
    """Compute ReLU activation,

        $max(x, 0)$
    :param x: pre-activations
    """
    # what's the issue here?

    return lab.smax(x, 0)


def logistic_fn(x: lab.Tensor) -> lab.Tensor:
    """Compute logistic activation,

        $1 / (1 + exp(-x))$
    :param x: pre-activations
    """
    return 1.0 / (1.0 + lab.exp(-x))
