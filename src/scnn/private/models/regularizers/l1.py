"""L1 Regularizer."""

from typing import Optional

import lab

from scnn.private.loss_functions import l1_penalty
from scnn.private.models.regularizers.regularizer import Regularizer


class L1Regularizer(Regularizer):
    """The l1-regularizer, which takes the form.

    .. math:: r(w) = \\lambda \\|w\\||_2^2.

    This regularizer is non-smooth at :math:`w = 0`.

    Attributes:
        lam: the regularization strength. Must be non-negative.
    """

    def penalty(self, w: lab.Tensor, **kwargs) -> float:
        """Compute the penalty associated with the regularizer.

        Args:
            w: parameter at which to compute the penalty.
        """

        return l1_penalty(w, self.lam)

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the minimum-norm sub-gradient of the l1 regularizer.

        Args:
            w: parameter at which to compute the penalty gradient.
            base_grad: the smooth component of the gradient, coming from the objective function.

        Returns:
            The minimum-norm subgradient.
        """
        indicators = w != 0

        smooth_term = lab.sign(w) * self.lam * indicators

        non_smooth_term = (
            lab.sign(base_grad)
            * lab.smin(lab.abs(base_grad), self.lam)
            * lab.logical_not(indicators)
        )

        return smooth_term - non_smooth_term
