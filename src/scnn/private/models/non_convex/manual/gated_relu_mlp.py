"""Implementation of one-layer Gated ReLU MLP with squared-error objective."""

from typing import Optional

import lab

from .relu_mlp import ReLUMLP
from scnn.private.models.regularizers import Regularizer
from scnn.private.loss_functions import squared_error, relu


class GatedReLUMLP(ReLUMLP):

    """One-layer Gated ReLU MLP with squared-error objective."""

    def __init__(
        self,
        d: int,
        U: lab.Tensor,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ):
        """
        :param d: the dimensionality of the dataset (ie.number of features).
        :param U: the gate vectors associated with gated ReLU activations.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        :param c: (optional) the number of targets.
        """

        super().__init__(d, U.shape[1], regularizer, c=c)
        self.U = U

    def _forward(self, X: lab.Tensor, w: lab.Tensor, **kwargs) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        """
        w1, w2 = self._split_weights(w)

        return lab.multiply(lab.sign(relu(X @ self.U)), X @ w1.T) @ w2.T

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute objective associated with examples X and targets y.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X, y)).
        """
        return squared_error(self._forward(X, w), y) / (
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
        """Compute the gradient of the l2 objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: parameter at which to compute the gradient pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        w1, w2 = self._split_weights(w)
        D = lab.sign(relu(X @ self.U))
        Z = lab.multiply(D, X @ w1.T)
        residuals = Z @ w2.T - y

        return self._grad_helper(X, residuals, w1, w2, D, Z) / self._scaling(
            y, scaling
        )

    def _grad_helper(
        self,
        X: lab.Tensor,
        residuals: lab.Tensor,
        w1: lab.Tensor,
        w2: lab.Tensor,
        D: lab.Tensor,
        Z: lab.Tensor,
    ) -> lab.Tensor:

        # always uses the 0 subgradient of max(x, 0) when x = 0.
        g1 = lab.einsum("ij, il, ik, lj->jk", D, residuals, X, w2)
        g2 = Z.T @ residuals

        return self._join_weights(g1, g2.T)
