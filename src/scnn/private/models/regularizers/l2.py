"""L2 Regularizer."""
from typing import Optional

import lab

from scnn.private.loss_functions import l2_penalty
from scnn.private.models.regularizers.regularizer import Regularizer


class L2Regularizer(Regularizer):

    """L2-regularizer of the form.

    $f(w) = (lambda/2) * ||w||_2^2$
    """

    def penalty(self, w: lab.Tensor, **kwargs) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty.
        :returns: penalty value
        """

        return l2_penalty(w, self.lam)

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the regularizer.

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: NOT USED. Gradient from the base model.
        :returns: gradient.
        """

        return self.lam * w
