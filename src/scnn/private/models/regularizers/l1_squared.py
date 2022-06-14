"""L2 Regularizer."""
from typing import Optional

import lab

from scnn.private.loss_functions import l1_squared_penalty
from scnn.private.models.regularizers.regularizer import Regularizer


class L1SquaredRegularizer(Regularizer):

    """L1-squared regularizer."""

    def penalty(self, w: lab.Tensor, **kwargs) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty.
        :returns: penalty value
        """

        return l1_squared_penalty(w, self.lam)
