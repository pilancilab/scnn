"""Convex formulation of two-layer Gated ReLU model with skip connections."""

from typing import Optional, Tuple, List, Callable

import lab

from scnn.private.models.regularizers import Regularizer
from scnn.private.models.convex import operators
from scnn.private.models.convex.convex_mlp import ConvexMLP
from scnn.private.loss_functions import squared_error


class SkipMLP(ConvexMLP):
    """Convex formulation of a two-layer MLP with gated ReLU activations
    and a linear skip connection."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U: Optional[lab.Tensor] = None,
        U_fn: Optional[Callable] = None,
        kernel: str = operators.EINSUM,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
        D_test: Optional[lab.Tensor] = None,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U: array of hyperplanes creating the sign patterns.
            Either U or U_fn must not be None.
        :param U_fn: function giving matrix of possible sign patterns.
            Either U or U_fn must not be None.
        :param kernel: the kernel to drive the matrix-vector operations.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        :param D_test: array of possible sign patterns for test data patterns.
        """
        super().__init__(d, D, U, U_fn, kernel, regularizer, c, D_test)

        self.weights = lab.zeros((self.c, self.p + 1, self.d))

    def set_weights(self, weights: lab.Tensor):
        if weights.shape == (self.c, self.p + 1, self.d):
            self.weights = weights
        elif weights.shape == (self.c, self.p, self.d):
            self.weights[:, :-1] = weights
        elif weights.shape == (self.c, 1, self.d):
            self.weights[:, -1:] = weights
        else:
            raise ValueError(
                f"Weights with shape {weights.shape} cannot be set to ConvexLassoNet with weight shape {(self.p + 2, self.d)}."
            )

    def _split_weights(self, w: lab.Tensor) -> Tuple[lab.Tensor, lab.Tensor]:

        # separate out positive and negative skip weights.
        return w[:, : self.p], w[:, self.p]

    def _join_weights(self, network_w: lab.Tensor, skip_w: lab.Tensor) -> lab.Tensor:

        return lab.concatenate([network_w, skip_w], axis=1)

    def get_weights(self) -> List[lab.Tensor]:
        """Get model weights in an interpretable format.

        :returns: list of tensors -- [network weights, skip weights].
        """

        return self._split_weights(self.weights)

    def get_reduced_weights(self) -> lab.Tensor:
        return self.weights

    def _forward(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :returns: predictions for X.
        """
        network_w, skip_w = self._split_weights(w.reshape(self.c, self.p + 1, self.d))

        return super()._forward(X, network_w, self._signs(X, D)) + X @ skip_w.T

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights *and* the
        L1 penalty on the skip connections.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """
        w = w.reshape(self.c, self.p + 1, self.d)

        return squared_error(self._forward(X, w, D), y) / (
            2 * self._scaling(y, scaling)
        )

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        flatten: bool = False,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the l2 objective with respect to the model
        weights.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        w = w.reshape(self.c, self.p + 1, self.d)
        D = self._signs(X, D)
        residual = self._forward(X, w, D) - y
        network_grad = lab.einsum("ij, il, ik->ljk", D, residual, X) / self._scaling(
            y, scaling
        )
        skip_grad = X.T @ residual / self._scaling(y, scaling)
        skip_grad = lab.expand_dims(skip_grad.T, axis=1)

        grad = self._join_weights(
            network_grad,
            skip_grad,
        )

        if flatten:
            grad = lab.ravel(grad)

        return grad
