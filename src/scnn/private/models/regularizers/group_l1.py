"""Group L1 Regularizer."""

from typing import Optional

import lab

from scnn.private.loss_functions import group_l1_penalty
from scnn.private.models.regularizers.regularizer import Regularizer


class GroupL1Regularizer(Regularizer):

    """The group-sparsity inducing Group L1-regularizer.

    The group-L1 regularizer, sometimes called the L1-L2 regularizer,
    has the mathematical form,

        ::math.. R(w) = \\lambda \\sum_{i \\in \\mathcal{I} ||w_i||_2,

    where :math:`\\mathcal{I}` is collection of disjoint index sets specifying
    the groups for the regularizer. This class expects the final axis of the
    inputs/weights :math:`w` to be the group axis.

    The group-L1 regularizer induces group-sparsity, meaning entire groups
    :math:`w_i` will be set to zero when :math:`\\lambda` is sufficiently
    large.

    Attributes:
        lam: the regularization strength.

    """

    def __init__(self, lam: float):
        """
        lam: a tuning parameter controlling the regularization strength.
        """

        self.lam = lam

    def penalty(
        self,
        w: lab.Tensor,
        **kwargs,
    ) -> float:
        """Compute the penalty associated with the regularizer.

        Args:
            w: parameter at which to compute the penalty.

        Returns:
            The value of the penalty at w.
        """
        return group_l1_penalty(w, self.lam)

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the minimum-norm subgradient (aka, the pseudo-gradient).

        Args:
            w: parameter at which to compute the penalty gradient.
            base_grad: the gradient of the un-regularized objective.
                This is required to compute the minimum-norm subgradient.

        Returns:
            minimum-norm subgradient.
        """
        # requires base_grad to compute minimum-norm subgradient
        assert base_grad is not None

        weight_norms = lab.sqrt(lab.sum(w ** 2, axis=-1, keepdims=True))
        grad_norms = lab.sqrt(lab.sum(base_grad ** 2, axis=-1, keepdims=True))

        # TODO: use safe divide
        non_smooth_term = (
            base_grad
            * lab.smin(self.lam / grad_norms, 1)
            * (weight_norms == 0)
        )
        smooth_term = self.lam * lab.safe_divide(w, weight_norms)

        # match input shape
        subgrad = smooth_term - non_smooth_term

        return subgrad
