"""Proximal operators. This module provides functions for solving minimization
problems of the form.

    $argmin_x { d(x,w) + beta * g(x) }$,
where d is a metric, g is a "simple" function, and beta is a parameter controlling the trade-off between d and g.

TODO:
    - Add proximal operator for L2-squared penalty so that we can support this using R-FISTA.
    - Clean-up group-by-feature to be less ugly.
    - Should we retain the Orthant and GroupL1Orthant operators? They don't have a use at the moment.
"""

from typing import Optional
import math

import lab


class ProximalOperator:

    """Base class for proximal operators."""

    def __call__(self, w: lab.Tensor, beta: Optional[float] = None) -> lab.Tensor:
        """Evaluate the proximal_operator.

        :param w: parameters to which apply the operator will be applied.
        :param beta: the coefficient in the proximal operator. This is usually a step-size.
        :returns: prox(w)
        """

        raise NotImplementedError("A proximal operator must implement '__call__'!")


class Identity(ProximalOperator):
    """The proximal-operator for the zero-function.

    This proximal-operator always returns the input point.
    """

    def __call__(self, w: lab.Tensor, beta: Optional[float] = None) -> lab.Tensor:
        """Evaluate the identity operator.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step. NOT USED.

        Returns:
            w, the original input point.
        """

        return w


class Regularizer(ProximalOperator):

    """Base class for proximal operators based on regularizers.

    Attributes:
        lam: the regularization strength. This must be non-negative.
    """

    def __init__(self, lam: float):
        """Initialize the proximal operator.

        Args:
            lam: a non-negative parameter controlling the regularization strength.
        """
        self.lam = lam


class L2(Regularizer):
    """The proximal operator for the squared l2-norm.

    The proximal operator returns the unique solution to the following optimization problem:

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\frac{\\beta * \\lambda}{2} \\|x\\|_2^2\\}.

    The solution is the shrinkage operator :math:`x^* = (1 + \\beta)^{-1} w`.
    """

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """

        return w / (1 + beta * self.lam)


class L1(Regularizer):

    """The proximal operator for the l1-norm.

    The l1 proximal operator is sometimes known as the soft-thresholding operator and
    is the unique solution to the following problem:

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\beta * \\lambda \\|x\\|_1\\}.
    """

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """

        return lab.sign(w) * lab.smax(lab.abs(w) - beta * self.lam, 0.0)


class GroupL1(Regularizer):

    """The proximal operator for the group-l1 regularizer.

    Given group indices :math:`\\calI`, the group-l1 regularizer is the sum of
    l2-norms of the groups, :math:`r(w) = \\sum_{i \\in \\calI} \\|w_i\\|_2`.
    The proximal operator is thus the unique solution to the following problem,

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\beta * \\lambda \\sum_{i=1 \\in \\calI} \\|x_i\\|_2\\}.

    Groups are either defined to be the last axis of the point w.
    """

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """
        # compute the squared norms of each group.
        norms = lab.sqrt(lab.sum(w**2, axis=-1, keepdims=True))

        w_plus = lab.multiply(
            lab.safe_divide(w, norms), lab.smax(norms - self.lam * beta, 0)
        )

        return w_plus


class FeatureGroupL1(Regularizer):

    """The proximal operator for the feature-wise group-l1 regularizer.

    The feature-wise group-l1 regularizer is the sum of l2-norms of the
    weights for each feature,

    ..math:: r(w) = \\sum_{i \\in \\calI} \\|w_i\\|_2.

    The proximal operator is the unique solution to the following problem,

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\beta * \\lambda
        \\sum_{j=1}^d \\|x_{:,j}\\|_2\\},

    where :math:`x_{:, j}` is the vector of all weights corresponding to
    feature :math:`j`.
    """

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """
        # compute the squared norms of each feature group.
        scaling = math.sqrt(w.shape[-1])

        norms = lab.sqrt(
            lab.sum(
                w**2,
                axis=tuple(range(len(w.shape) - 1)),
                keepdims=True,
            )
        )

        w_plus = lab.multiply(
            lab.safe_divide(w, norms),
            lab.smax(norms - scaling * self.lam * beta, 0),
        )

        return w_plus


# ==============
# REFACTOR BELOW
# ==============


class Orthant(ProximalOperator):

    """The projection operator for the orthant constrain, A_i x >= 0, where A_i
    is a diagonal matrix with (A_i)_jk in {-1, 1}."""

    def __init__(self, A: lab.Tensor):
        """
        A: a matrix of sign patterns defining orthants on which to project.
            The diagonal A_i is stored as the i'th column of A.
        """
        self.A = A
        if len(A.shape) == 3:
            self.sum_string = "ikj,imjk->imjk"
        else:
            self.sum_string = "kj,imjk->imjk"

    def __call__(self, w: lab.Tensor, beta: Optional[float] = None) -> lab.Tensor:
        """
        w: parameters to which the projection will be applied.
            This should be a (k x c x p x d) array, where each element of axis -1 corresponds to one column of A.
        beta: NOT USED. The coefficient in the proximal operator. This is usually a step-size.
        :returns: updated parameters.
        """

        return lab.where(
            lab.einsum(self.sum_string, self.A, w) >= 0, w, lab.zeros_like(w)
        )


class GroupL1Orthant(Regularizer):

    """Mixed group L1 penalty with orthant constraint."""

    def __init__(self, d: int, lam: float, A: lab.Tensor):
        """
        d: the dimensionality of each group for the regularizer.
        lam: the strength of the group-L1 regularizer.
        A: a matrix of sign patterns defining orthants on which to project.
            The diagonal A_i is stored as the i'th column of A.
        """
        self.d = d
        self.A = A
        self.lam = lam

        self.group_prox = GroupL1(lam)
        self.orthant_proj = Orthant(A)

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """
        w: parameters to which the projection will be applied.
            This should be a (k x c x p x d + n) array.
            The first 'd' entries in element of axis -1 correspond to the model weights (with group L1 regularizer)
            and the remaining 'n' entries correspond to the slack variables (with orthant constraint).
        beta: The coefficient in the proximal operator. This is usually a step-size.
        :returns: updated parameters.
        """
        model_weights, slacks = w[:, :, :, : self.d], w[:, :, :, self.d :]
        w_plus, s_plus = self.group_prox(model_weights, beta), self.orthant_proj(
            slacks, beta
        )

        return lab.concatenate([w_plus, s_plus], axis=-1)
