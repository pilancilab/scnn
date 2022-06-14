"""Interface for model classes."""

from typing import Optional, Union

import lab


class Regularizer:

    """Base class for regularizers."""

    lam: float

    def __init__(
        self,
        lam: float,
    ):
        """
        :param base_model: the Model instance to regularize.
        :param lam: the tuning parameter controlling the strength of regularization.
        """

        self.lam = lam

    # abstract methods that must be overridden

    def penalty(self, w: lab.Tensor, **kwargs) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty value.
        :returns: penalty value
        """

        raise NotImplementedError("A regularizer must implement 'penalty'!")

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        **kargs,
    ) -> lab.Tensor:
        """Compute the gradient of the regularizer.

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: (optional) the gradient of the un-regularized objective. This is
            used to compute the minimum-norm subgradient for "pseudo-gradient" methods.
        :returns: gradient.
        """

        raise NotImplementedError("A regularizer must implement 'grad'!")
