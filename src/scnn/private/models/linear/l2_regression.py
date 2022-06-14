"""Implementation of linear regression with the least-squares objective."""
from typing import Optional

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
import scnn.private.loss_functions as loss_fns


class LinearRegression(Model):

    """Linear regression with least-squares objective.

    Attributes:
        d: the input dimension.
        c: the output dimension.
        weights: an (c, d) matrix of weights for the linear model.
    """

    def __init__(
        self, d: int, c: int = 1, regularizer: Optional[Regularizer] = None
    ):
        """
        Args:
            d: the input dimension.
            c: the output dimension.
            regularizer: a penalty function controlling the flexibility of the
                model.
        """
        super().__init__(regularizer)
        self.d = d
        self.c = c
        self.weights = lab.zeros((c, d))

    def _forward(self, X: lab.Tensor, w: lab.Tensor, **kwargs) -> lab.Tensor:
        """Compute forward pass.

        Args:
            X: (n,d) array containing the data examples.
            w: parameter at which to compute the forward pass.

        Returns:
            The forward pass (i.e. predictions) at `w`.
        """
        return X @ w.T

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute least-squares objective given a dataset.

        Args:
            X: (n,d) array containing the data examples.
            y: (n,c) array containing the data targets.
            w: specific parameter at which to compute the forward pass.
            scaling: scaling parameter for the objective. Defaults to `n * c`.

        Returns:
            The objective at `w`.
        """
        return loss_fns.squared_error(self._forward(X, w), y) / (
            2 * self._scaling(y, scaling)
        )

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute gradient of the least-squares objective.

        Args:
            X: (n,d) array containing the data examples.
            y: (n,c) array containing the data targets.
            w: parameter at which to compute the forward pass.
            scaling: scaling parameter for the objective. Defaults to `n * c`.

        Returns:
            The gradient at `w`.
        """

        res = self._forward(X, w) - y
        grad = lab.matmul(res.T, X) / self._scaling(y, scaling)
        return grad
