"""Group L1 Regularizer."""
from typing import Optional
import math

import lab

from scnn.private.models.regularizers.regularizer import Regularizer


class FeatureGroupL1Regularizer(Regularizer):

    """The group-sparsity inducing Group L1-regularizer over features.

    The group-L1 regularizer, sometimes called the L1-L2 regularizer,
    has the mathematical form,

        ::math.. R(w) = \\lambda \\sum_{i \\in \\mathcal{I} ||w_i||_2,

    where :math:`\\mathcal{I}` is collection of disjoint index sets specifying
    the groups for the regularizer. This class expects the second-to-last axis
    of the inputs/weights :math:`w` to be the feature-wise axis defining the
    different groups.

    This regularizer induces group-sparsity, meaning entire features will be
    dropped out of the model when :math:`\\lambda` is sufficiently large.

    Attributes:
        lam: the regularization strength.

    """

    def __init__(self, lam: float):
        """
        Args:
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
        scaling = math.sqrt(w.shape[-1])

        feature_norms = (
            scaling
            * self.lam
            * lab.sqrt(
                lab.sum(
                    w ** 2,
                    axis=tuple(range(len(w.shape) - 1)),
                )
            )
        )

        return lab.sum(feature_norms)

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the minimum-norm subgradient (aka, the pseudo-gradient).

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: the gradient of the un-regularized objective. This is required
            to compute the minimum-norm subgradient.
        :returns: minimum-norm subgradient.
        """
        # requires base_grad to compute minimum-norm subgradient
        assert base_grad is not None
        scaling = math.sqrt(w.shape[-1])

        dims_to_sum = tuple(range(len(w.shape) - 1))
        weight_norms = lab.sqrt(
            lab.sum(
                w ** 2,
                axis=dims_to_sum,
                keepdims=True,
            )
        )
        grad_norms = lab.sqrt(
            lab.sum(
                base_grad ** 2,
                axis=dims_to_sum,
                keepdims=True,
            )
        )

        non_smooth_term = (
            base_grad
            * lab.smin(scaling * self.lam / grad_norms, 1)
            * (weight_norms == 0)
        )
        smooth_term = scaling * self.lam * lab.safe_divide(w, weight_norms)

        # match input shape
        subgrad = smooth_term - non_smooth_term

        return subgrad
