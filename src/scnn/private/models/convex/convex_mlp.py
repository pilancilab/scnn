"""Convex formulations of neural networks."""

from typing import Optional, List, Tuple, Dict
from math import ceil

from scipy.sparse.linalg import LinearOperator  # type: ignore

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
from . import operators
from scnn.private.utils import MatVecOperator
from scnn.private.loss_functions import squared_error, relu

# two-layer MLPs with ReLU activations.


class ConvexMLP(Model):
    """Convex formulation of a two-layer neural network (multi-layer
    perceptron) with ReLU activations."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U: lab.Tensor,
        kernel: str = operators.EINSUM,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U: array of hyperplanes creating the sign patterns.
        :param kernel: the kernel to drive the matrix-vector operations.
        """
        super().__init__(regularizer)

        self.d = d
        self.p = D.shape[1]  # each column is a unique sign pattern
        self.c = c
        self.weights = lab.zeros(
            (c, self.p, self.d)
        )  # one linear model per sign pattern

        self.D = D
        self.U = U
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

    def _weights(self, w: Optional[lab.Tensor]) -> lab.Tensor:
        return self.weights if w is None else w

    def get_reduced_weights(self) -> lab.Tensor:
        return self.weights

    def _signs(self, X: lab.Tensor, D: Optional[lab.Tensor] = None):
        local_D = self.D
        if D is not None:
            return D
        elif not self._train:
            local_D = relu(X @ self.U)
            local_D[local_D > 0] = 1

        return local_D

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
        return self._data_mvp(w, X, self._signs(X, D))

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """

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
        weights. As in 'self.__call__' above, we could optimize this by
        implementing it in a faster low-level language.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        grad = self._gradient(w, X, y, self._signs(X, D)) / self._scaling(
            y, scaling
        )

        if flatten:
            grad = grad.reshape(-1)

        return grad

    def data_operator(
        self,
        X: lab.Tensor,
        D: Optional[lab.Tensor] = None,
    ) -> LinearOperator:
        """Construct a matrix operator to evaluate the matrix-vector equivalent
        to the sum,

        .. math::

            \\sum_i D_i X v_i

        where v_i is the i'th block of the input vector 'v'. This is equivalent to
        constructing the expanded matrix :math:`A = [D_1 X, D_2 X, ..., D_P X]` and then evaluating :math:`Av`.
        Use 'data_matrix' to directly compute $A$.
        :param X: (n,d) array containing the data examples.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :returns: LinearOperator which computes the desired product.
        """
        n, _ = X.shape
        pd = self.d * self.p
        local_D = self._signs(X, D)

        # pre-compute extended matrix
        if self.kernel == operators.DIRECT:
            expanded_X = self._data_builder(X, local_D)

            def forward(v):
                return lab.squeeze(
                    self._data_mvp(v, X=X, D=local_D, expanded_X=expanded_X)
                )

            def transpose(v):
                return lab.squeeze(
                    self._data_t_mvp(v, X=X, D=local_D, expanded_X=expanded_X)
                )

        else:

            def forward(v):
                return lab.squeeze(self._data_mvp(v, X=X, D=local_D))

            def transpose(v):
                return lab.squeeze(self._data_t_mvp(v, X=X, D=local_D))

        op = MatVecOperator(
            shape=(n, pd), forward=forward, transpose=transpose
        )

        return op

    def add_new_patterns(
        self, patterns: lab.Tensor, weights: lab.Tensor, remove_zero=False
    ) -> lab.Tensor:
        """Attempt to augment the current model with additional sign patterns.

        :param patterns: the tensor of sign patterns to add to the current set.
        :param weights: the tensor of weights which induced the new patterns.
        :param remove_zero: whether or not to remove the zero vector.
        :returns: None
        """

        if lab.size(patterns) > 0:
            # update neuron count.
            self.D = lab.concatenate([self.D, patterns], axis=1)
            self.U = lab.concatenate([self.U, weights.T], axis=1)

            # initialize new model components at 0.
            added_weights = lab.zeros((self.c, weights.shape[0], self.d))
            self.weights = lab.concatenate(
                [self.weights, added_weights], axis=1
            )

            # filter out the zero column.
            if remove_zero:
                non_zero_cols = lab.logical_not(
                    lab.all(self.D == lab.zeros((self.D.shape[0], 1)), axis=0)
                )
                self.D = self.D[:, non_zero_cols]
                self.U = self.U[:, non_zero_cols]
                self.weights = self.weights[:, non_zero_cols]

            self.p = self.D.shape[1]

        return added_weights

    def batch_X(
        self, batch_size: Optional[int], X: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:
        D = self._signs(X)

        if batch_size is None:
            return [{"X": X, "D": D}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "D": D[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]

    def batch_Xy(
        self, batch_size: Optional[int], X: lab.Tensor, y: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:
        D = self._signs(X)

        if batch_size is None:
            return [{"X": X, "y": y, "D": D}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "y": y[i * batch_size : (i + 1) * batch_size],
                "D": D[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]

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

        return lab.sign(relu(X @ self.U)), self.U.T
