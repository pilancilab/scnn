"""Combined group L1 penalty and orthant constraints for the augmented
Lagrangian."""
from typing import Optional, Union, Callable

import lab

from scnn.private.models.regularizers.orthant import OrthantConstraint
from scnn.private.models.regularizers.group_l1 import GroupL1Regularizer

from scnn.private.prox import Orthant


class GroupL1Orthant(GroupL1Regularizer, OrthantConstraint):

    """Combined group L1 and orthant constraint for the augmented Lagrangian of
    a convex neural network."""

    def __init__(self, A: lab.Tensor, lam: float):
        """
        :param A: a matrix of sign patterns defining orthants on which to project.
            The diagonal A_i is stored as the i'th column of A.
        :param lam: the tuning parameter controlling the strength of regularization.
        """

        self.orthant_proj = Orthant(A)
        self.lam = lam

    def penalty(  # type: ignore
        self,
        w: lab.Tensor,
        split_weights: Callable[[lab.Tensor], lab.Tensor],
        **kwargs,
    ) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty.
        :param split_weights: a function which can be called to split the weights
            into the group l1 terms and the orthant-constrained terms.
        :returns: penalty value
        """
        model_weights, _ = split_weights(w)

        return super().penalty(model_weights)

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        split_weights: Callable[[lab.Tensor], lab.Tensor] = None,
        merge_weights: Callable[[lab.Tensor, lab.Tensor], lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the objective with respect to the model
        parameters.

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: (optional) the gradient of the un-regularized objective. This is
            used to compute the minimum-norm subgradient for "pseudo-gradient" methods.
        :param split_weights: a function which can be called to split the weights
            into the group l1 terms and the orthant-constrained terms.
        :param merge_weights: a function which can be called to merge the
            group-l1 germs and the orthant constrained terms into one parameter list.
        :returns: the gradient
        """
        assert base_grad is not None
        assert split_weights is not None
        assert merge_weights is not None

        model_weights, slacks = split_weights(w)
        model_weights_grad, slacks_grad = split_weights(base_grad)

        # compute min-norm subgradient for group-1 penalized variables.
        # super might not be reliable!
        min_norm_subgrad = model_weights_grad + super().grad(
            model_weights_grad, model_weights
        )

        # compute gradient mapping for orthant constrained variables.
        gradient_mapping = self._gradient_mapping(slacks, slacks_grad) - slacks_grad

        return merge_weights(min_norm_subgrad, gradient_mapping)
