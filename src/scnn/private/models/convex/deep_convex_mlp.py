"""Convex formulations of neural networks."""

from typing import Optional, List, Tuple, Dict, Callable
from math import ceil

from scipy.sparse.linalg import LinearOperator  # type: ignore

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
from . import operators
from .convex_mlp import ConvexMLP
from scnn.private.utils import MatVecOperator
from scnn.private.loss_functions import squared_error, relu

# two-layer MLPs with ReLU activations.


class DeepConvexMLP(ConvexMLP):
    """Convex formulation of a multi-layer neural network with
    ReLU activations."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U_fn: Callable,
        kernel: str = operators.EINSUM,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
        D_test: Optional[lab.Tensor] = None,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U_fn: function giving matrix of possible sign patterns.
        :param kernel: the kernel to drive the matrix-vector operations.
        :param D_test: array of possible sign for test data patterns.
        """
        self.regularizer = regularizer

        self.d = d
        self.p = D.shape[1]  # each column is a unique sign pattern
        self.c = c
        self.weights = lab.zeros(
            (c, self.p, self.d)
        )  # one linear model per sign pattern

        self.D = D
        self.D_test = D_test
        self.U_fn = U_fn
        self.kernel = kernel

        (
            self._data_mvp,
            self._data_t_mvp,
            self._gradient,
            self._hessian_mvp,
            self._bd_hessian_mvp,
        ) = operators.get_kernel(kernel)

        (
            self._data_builder,
            self._hessian_builder,
            self._bd_hessian_builder,
        ) = operators.get_matrix_builders(kernel)

        self._train = True

    def _signs(self, X: lab.Tensor, D: Optional[lab.Tensor] = None):
        local_D = self.D

        if D is not None:
            return D

        if not self._train:
            if self.D_test is not None:
                local_D = self.D_test
            else:
                local_D = lab.tensor(self.U_fn(lab.to_np(X)))

        return local_D
