"""Augmented lagrangian for the two-layer MLP with cone constraints."""
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import lab

from scnn.private.loss_functions import squared_error
from scnn.private.models.convex import operators
from scnn.private.models.convex.convex_mlp import ConvexMLP
from scnn.private.models.regularizers import Regularizer


class AL_MLP(ConvexMLP):

    """Augmented Lagrangian for the two-layer ReLU MLP training problem."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U: lab.Tensor,
        kernel: str = operators.EINSUM,
        delta: float = 100.0,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U: array of hyperplanes creating the sign patterns.
        :param c: number of classes for this problem
        :param kernel: the kernel to drive the matrix-vector operations.
        :param delta: parameter controlling the strength of the quadratic penalty
        in the augmented Lagrangian.
        """

        super().__init__(d, D, U, kernel=kernel, regularizer=regularizer, c=c)
        self.n = self.D.shape[0]
        self.c = c

        self.weights = lab.zeros((2, self.c, self.p, self.d))

        # estimates of the optimal dual variables
        self.e_multipliers = lab.tensor([0.0])  # NOT USED.
        self.i_multipliers = lab.zeros((2, self.c, self.p, self.n))

        self.delta = delta / self.n
        self.orthant = 2 * self.D - lab.ones_like(self.D)

    def _orthant(
        self, index_range: Optional[Tuple[int, int]] = None
    ) -> lab.Tensor:
        if index_range is None:
            return self.orthant

        return self.orthant[index_range[0] : index_range[1]]

    def _i_multipliers(
        self, index_range: Optional[Tuple[int, int]] = None
    ) -> lab.Tensor:
        if index_range is None:
            return self.i_multipliers

        return self.i_multipliers[:, :, :, index_range[0] : index_range[1]]

    def update_model_weights(self, w: lab.Tensor):
        self.weights = w

    def set_weights(self, weights: lab.Tensor):
        if weights.shape == (2, self.c, self.p, self.d):
            self.weights = weights
        elif weights.shape == (self.c, self.p, self.d):
            # "split" the weights and assign to the model
            self.weights = lab.stack([weights / 2, -weights / 2])
        else:
            raise ValueError(
                f"Weights with shape {weights.shape} cannot be set to IneqAugmentedConvexMLP with weight shape {self.weights.shape}."
            )

    def get_reduced_weights(self) -> lab.Tensor:
        return self.weights[0] - self.weights[1]

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

        return super()._forward(X, w[0] - w[1], D)

    def lagrange_penalty_objective(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute the penalty term in the augmented Lagrangian.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the objective.
        :returns: the objective
        """
        w = w.reshape(2, self.c, self.p, self.d)

        gap = self.i_constraint_gap(X, w, index_range)
        scaled_multipliers = lab.safe_divide(
            self._i_multipliers(index_range), self.delta
        )

        penalty = (
            self.delta
            * lab.sum(lab.smax(gap + scaled_multipliers, 0) ** 2)
            / 2
        )

        return penalty

    def lagrange_penalty_grad(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        flatten=False,
        **kwargs,
    ) -> Union[Tuple[lab.Tensor, lab.Tensor], lab.Tensor]:
        """Compute the gradient of the penalty term in the augmented
        Lagrangian.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the gradient.
        :param flatten: whether or not to flatten the output into a single vector.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        w = w.reshape(2, self.c, self.p, self.d)
        gap = self.i_constraint_gap(X, w, index_range)

        scaled_multipliers = lab.safe_divide(
            self._i_multipliers(index_range),
            self.delta,
        )
        shifted_gap = lab.smax(gap + scaled_multipliers, 0)
        grad = -(
            self.delta
            * lab.einsum(
                "ij, lmji, ik->lmjk",
                self._orthant(index_range),
                shifted_gap,
                X,
            )
        )

        if flatten:
            return lab.ravel(grad)

        return grad

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        ignore_lagrange_penalty: bool = False,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: specific parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """
        obj = squared_error(self._forward(X, w, D), y) / (
            2 * self._scaling(y, scaling)
        )
        if not ignore_lagrange_penalty:
            obj += self.lagrange_penalty_objective(X, w, index_range, scaling)

        return obj

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        ignore_lagrange_penalty: bool = False,
        flatten: bool = False,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of augmented Lagrangian objective with respect
        to the model weights.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :returns: the gradient
        """
        w = w.reshape(2, self.c, self.p, self.d)
        combined_weights = w[0] - w[1]
        v_grad = super()._grad(X, y, combined_weights, D, scaling=scaling)

        grad = lab.stack([v_grad, -v_grad])
        if not ignore_lagrange_penalty:
            grad = grad + self.lagrange_penalty_grad(
                X, w, D=D, index_range=index_range, scaling=scaling
            )

        if flatten:
            return lab.ravel(grad)

        return grad

    def lagrangian(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        ignore_lagrange_penalty: bool = False,
        **kwargs,
    ) -> float:
        """Compute the Lagrangian function.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: specific parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """
        w = self._weights(w).reshape(2, self.c, self.p, self.d)

        # doesn't include regularization
        obj = squared_error(self._forward(X, w, D), y) / (
            2 * self._scaling(y, scaling)
        )

        # regularization
        if self.regularizer is not None:
            obj += self.regularizer.penalty(w)

        gap = self.i_constraint_gap(X, w, index_range)

        # penalty terms from Lagrangian
        penalty = lab.sum(gap * self._i_multipliers(index_range))

        return obj + penalty

    def lagrangian_grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        flatten: bool = False,
    ) -> lab.Tensor:
        w = self._weights(w).reshape(2, self.c, self.p, self.d)

        # Doesn't include regularization.
        obj_grad = self._grad(
            X,
            y,
            w,
            D,
            index_range,
            scaling,
            ignore_lagrange_penalty=True,
        )

        # penalty terms from Lagrangian
        penalty_grad = lab.einsum(
            "imjk, kj, kl -> imjl",
            self._i_multipliers(index_range),
            self._orthant(index_range),
            X,
        )

        grad = obj_grad - penalty_grad

        # regularization
        if self.regularizer is not None:
            grad += self.regularizer.grad(w, grad)

        if flatten:
            return lab.ravel(grad)

        return grad

    def i_constraint_gap(
        self,
        X: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
    ):
        """Compute violation of the linear constraints.

        .. math::

            X v_i - a_i

            X w_i - b_i,

        where a_i, b_i are the slack variables.
        :param X: (n,d) array containing the data examples.
        :param w: (optional) specific parameter at which to compute the gradient.
        Defaults to 'None', in which case the current model state is used.
        :returns: the constraint gaps.
        """
        weights = self._weights(w).reshape(2, self.c, self.p, self.d)
        gaps = -lab.einsum(
            "ij, lmkj, ik-> lmki", X, weights, self._orthant(index_range)
        )

        return gaps

    def constraint_gaps(
        self,
        X: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
    ):
        """Compute violation of the constraints.

        :param X: (n,d) array containing the data examples.
        :param w: (optional) specific parameter at which to compute the gradient.
        Defaults to 'None', in which case the current model state is used.
        :returns: equality constraint gap, inequality constraint gap.
        """

        return lab.tensor([0.0]), lab.smax(
            self.i_constraint_gap(X, w, index_range), 0
        )

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
                "index_range": (i * batch_size, (i + 1) * batch_size),
            }
            for i in range(n_batches)
        ]

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
            added_weights = lab.zeros((2, self.c, weights.shape[0], self.d))
            self.weights = lab.concatenate(
                [self.weights, added_weights], axis=2
            )

            # update dual variables
            added_multipliers = lab.zeros(
                (2, self.c, weights.shape[0], self.n)
            )
            self.i_multipliers = lab.concatenate(
                [self.i_multipliers, added_multipliers], axis=2
            )

            # filter out the zero column.
            if remove_zero:
                non_zero_cols = lab.logical_not(
                    lab.all(self.D == lab.zeros((self.D.shape[0], 1)), axis=0)
                )
                self.D = self.D[:, non_zero_cols]
                self.U = self.U[:, non_zero_cols]
                self.weights = self.weights[:, :, non_zero_cols]
                self.i_multipliers = self.i_multipliers[:, :, non_zero_cols]

            self.p = self.D.shape[1]

        self.orthant = 2 * self.D - lab.ones_like(self.D)

        return added_weights
