"""Implementation of one-layer ReLU MLP with squared-error objective."""
from typing import Optional, Tuple

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
from scnn.private.loss_functions import squared_error, relu


class ReLUMLP(Model):

    """One-layer ReLU MLP with squared-error objective."""

    activation_history: Optional[lab.Tensor] = None
    weight_history: Optional[lab.Tensor] = None

    def __init__(
        self,
        d: int,
        p: int,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ):
        """
        :param d: the dimensionality of the dataset (ie.number of features).
        :param p: the number of hidden units.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        """
        super().__init__(regularizer)
        self.d = d
        self.p = p
        self.c = c

        # use random initialization by default.
        self.weights = lab.tensor(
            lab.np_rng.standard_normal((d * p + p * c)), dtype=lab.get_dtype()
        )

    def set_weights(self, weights: lab.Tensor):
        # weights include second layer
        if weights.shape == (self.d * self.p + self.p * self.c):
            self.weights = weights
        # weights consist only of first-layer weights.
        elif weights.shape == (self.d * self.p,):
            self.weights[: self.p * self.d] = weights
        # weights consist only of second-layer weights.
        elif weights.shape == (self.p * self.c,):
            self.weights[self.p * self.d :] = weights
        # weights need to be flattened.
        elif lab.size(weights) == (self.d * self.p + self.p * self.c):
            self.weights = lab.ravel(weights)
        else:
            raise ValueError(
                f"Weights with shape {weights.shape} cannot be set to ReluMLP with weight shape {self.d * self.p + p}."
            )

    def _split_weights(self, w: lab.Tensor) -> Tuple[lab.Tensor, lab.Tensor]:

        return (
            w[: self.d * self.p].reshape(self.p, self.d),
            w[self.d * self.p :].reshape(self.c, self.p),
        )

    def _join_weights(self, w1: lab.Tensor, w2: lab.Tensor) -> lab.Tensor:

        return lab.concatenate([lab.ravel(w1), lab.ravel(w2)])

    def _forward(self, X: lab.Tensor, w: lab.Tensor, **kwargs) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        """
        w1, w2 = self._split_weights(w)

        return relu(X @ w1.T) @ w2.T

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
        :param w: specific parameter at which to compute the forward pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X, y)).
        """
        return squared_error(self._forward(X, w), y) / (2 * self._scaling(y, scaling))

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
        Z = relu(X @ w1.T)
        D = lab.sign(Z)

        residuals = Z @ w2.T - y

        return self._grad_helper(X, residuals, w1, w2, D, Z) / self._scaling(y, scaling)

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
        g2 = residuals.T @ Z

        return self._join_weights(g1, g2)

    def sign_patterns(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> Tuple[lab.Tensor, lab.Tensor]:
        """Compute the gradient of the l2 objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: (optional) specific parameter at which to compute the sign patterns.
        :returns: the set of sign patterns active at w or the current models parameters if w is None.
        """

        w1, w2 = self._split_weights(self._weights(w))
        return lab.sign(lab.smax(lab.matmul(X, w1.T), 0)), w1
