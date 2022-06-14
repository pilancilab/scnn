"""Implementation of logistic regression for binary classification."""
from typing import Optional

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
import scnn.private.loss_functions as loss_fns


class LogisticRegression(Model):

    """Logistic regression with for binary classification."""

    def __init__(self, d: int, regularizer: Optional[Regularizer] = None):
        """
        :param d: the dimensionality of the dataset (ie.number of features).
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        """
        super().__init__(regularizer)

        self.weights = lab.zeros(d)

    def _forward(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        """

        return loss_fns.logistic_fn(X @ w)

    def _logits(self, X: lab.Tensor, w: lab.Tensor) -> lab.Tensor:
        return X @ w

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute logistic objective associated with examples X and targets y.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X,y)).
        """
        return loss_fns.logistic_loss(self._logits(X, w), y) / self._scaling(y, scaling)

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the logistic objective with respect to the
        model parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        logits = self._logits(X, w)
        return (
            -X.T
            @ lab.multiply(y, loss_fns.logistic_fn(-y * logits))
            / self._scaling(y, scaling)
        )
