"""Orthant constraint."""
from typing import Optional

import lab

from scnn.private.models.regularizers.regularizer import Regularizer

from scnn.private.prox import ProximalOperator


class Constraint(Regularizer):

    """Representation of a regularizer based on convex constraints of the form,
    f(W) geq 0, where f is a convex function and W are the model weights. In
    essence, this class is a wrapper for computation of the gradient mapping,

        g(xk) = xk - proj(xk - grad(xk)),
    which can be used as a stopping criterion.
    """

    lam = 0.0
    projection_op: ProximalOperator

    def __init__(self):
        raise NotImplementedError(
            "A constraint-based regularizer must implement its own constructor."
        )

    def _gradient_mapping(
        self,
        w: lab.Tensor,
        grad: lab.Tensor,
        step_size: float = 1.0,
    ) -> lab.Tensor:
        w_plus = w - step_size * grad
        return (w - self.projection_op(w_plus)) / step_size

    def penalty(self, w: lab.Tensor, **kwargs) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty.
        :returns: penalty value
        """
        return 0.0

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        step_size: float = 1.0,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the objective with respect to the model
        parameters.

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: gradient from the base model.
        :param step_size: step-size to use when computing the gradient mapping.
        :returns: the gradient
        """
        assert base_grad is not None
        return self._gradient_mapping(w, base_grad, step_size) - base_grad
