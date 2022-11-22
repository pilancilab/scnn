"""Group L1 Regularizer with L2 penalty on skip connections."""

from typing import Optional

import lab

from scnn.private.loss_functions import group_l1_penalty
from scnn.private.models.regularizers.regularizer import Regularizer
from scnn.private.models.regularizers.group_l1 import GroupL1Regularizer


class SkipGroupL1Regularizer(Regularizer):

    """The group-sparsity inducing Group L1-regularizer with L2 skip penalty.

    The group-L1 regularizer, sometimes called the L1-L2 regularizer,
    has the mathematical form,

        ::math.. R(w) = \\lambda \\sum_{i \\in \\mathcal{I} ||w_i||_2,

    where :math:`\\mathcal{I}` is collection of disjoint index sets specifying
    the groups for the regularizer. This class expects the final axis of the
    inputs/weights :math:`w` to be the group axis.

    Unlike :class:`GroupL1Regularizer <scnn.private.models.regularizers.GroupL1Regularizer`,
    SkipGroupL1Regularizer places a separate :math:`\\ell_2` penalty on
    the skip connections, if there are any.

    Attributes:
        lam: the regularization strength.
        skip_lam: the regularizer strength for the penalty on the skip connections.
    """

    def __init__(self, lam: float, skip_lam: float):
        """
        lam: a tuning parameter controlling the regularization strength
            for the group L1 penalty.
        lam: a tuning parameter controlling the regularization strength
            for the L2 penalty on the skip connections.
        """

        self.lam = lam
        self.group_l1 = GroupL1Regularizer(self.lam)
        self.skip_lam = skip_lam

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
        if len(w.shape) == 3:
            network_w, skip_w = w[:, :-1], w[:, -1:]
        else:
            network_w, skip_w = w[:, :, :-1], w[:, :, -1:]

        return group_l1_penalty(network_w, self.lam) + self.skip_lam * lab.sum(
            skip_w**2
        )

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

        if len(w.shape) == 3:
            network_w, skip_w = w[:, :-1], w[:, -1:]
            network_grad = base_grad[:, :-1]
            axis = 1
        else:
            network_w, skip_w = w[:, :, :-1], w[:, :, -1:]
            network_grad = base_grad[:, :, :-1]
            axis = 2

        network_subgrad = self.group_l1.grad(network_w, network_grad)
        skip_subgrad = self.skip_lam * skip_w

        subgrad = lab.concatenate([network_subgrad, skip_subgrad], axis=axis)

        return subgrad
