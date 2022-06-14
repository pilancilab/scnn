"""Approximate cone decompositions via one-sided quadratic.
"""
from typing import Optional, Tuple, List, Dict
from math import ceil

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer

import lab


class QuadraticDecomposition(Model):

    """Quadratic approximation to the cone decomposition program."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
        combined: bool = True,
    ) -> None:
        """
        Args:
            d: the dimensionality of the dataset (ie. number of features).
            D: array of possible sign patterns.
            c: number of classes for this problem.
        """

        super().__init__(regularizer=regularizer)

        self.d = d
        self.D = D
        self.n, self.p = self.D.shape
        self.c = c

        self.weights = lab.zeros((self.c, self.p, self.d))

        self.orthant = 2 * self.D - lab.ones_like(self.D)

        self.combined = combined

    def update_model_weights(self, w: lab.Tensor):
        self.weights = w

    def _orthant(
        self, index_range: Optional[Tuple[int, int]] = None
    ) -> lab.Tensor:
        if index_range is None:
            return self.orthant

        return self.orthant[index_range[0] : index_range[1]]

    def _forward(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        orthant: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.

        Args:
            X: (n,d) array containing the data examples.
            w: parameter at which to compute the forward pass.
            orthant:
        """

        weights = self._weights(w).reshape(self.c, self.p, self.d)

        return lab.einsum("ij, mkj, ik-> mki", X, weights, orthant)

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        orthant: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights.

        Args:
            X: (n,d) array containing the data examples.
            y: (n,d) array containing the data targets.
            w: specific parameter at which to compute the objective.
            orthant: (optional) specific activation matrix at which to compute the forward pass.
                Defaults to self.D or manual computation depending on the value of self._train.
            scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        """
        preds = self._forward(X, w, orthant)
        residual = lab.smax(y - preds, 0)
        obj = lab.sum(residual ** 2) / (2 * self._scaling(y, scaling))

        if not self.combined:
            obj += lab.sum(lab.smin(preds, 0) ** 2) / (
                2 * self._scaling(y, scaling)
            )

        return obj

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        orthant: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        flatten: bool = False,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of augmented Lagrangian objective with respect
        to the model weights.

        Args:
            X: (n,d) array containing the data examples.
            y: (n,d) array containing the data targets.
            w: specific parameter at which to compute the objective.
            orthant: (optional) specific activation matrix at which to compute the forward pass.
                Defaults to self.D or manual computation depending on the value of self._train.
            scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        """
        preds = self._forward(X, w, orthant)
        residual = lab.smax(y - preds, 0)

        grad = (
            -lab.einsum(
                "ij, mji, ik->mjk",
                orthant,
                residual,
                X,
            )
            / self._scaling(y, scaling)
        )

        if not self.combined:
            grad += (
                lab.einsum(
                    "ij, mji, ik->mjk",
                    orthant,
                    lab.smin(preds, 0),
                    X,
                )
                / self._scaling(y, scaling)
            )

        if flatten:
            return lab.ravel(grad)

        return grad

    def batch_X(
        self, batch_size: Optional[int], X: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:

        if batch_size is None:
            return [{"X": X, "orthant": self.orthant}]
        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "orthant": self.orthant[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]

    def batch_Xy(
        self, batch_size: Optional[int], X: lab.Tensor, y: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:
        orthant = self.orthant

        if batch_size is None:
            return [{"X": X, "y": y, "orthant": self.orthant}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "y": y[i * batch_size : (i + 1) * batch_size],
                "orthant": orthant[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]
