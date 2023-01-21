"""Group L1 Regularizer."""

from typing import Optional

import lab

from .group_l1 import GroupL1Regularizer


class CardinalityConstraint(GroupL1Regularizer):

    """Experimental group-L1 regularizer with additional sparsity constraints."""

    def __init__(self, lam: float, M: float, b: int):
        """
        lam: a tuning parameter controlling the regularization strength.
        M: magnitude of the weight constraints.
        b: maximum number of features allowed.
        """
        super().__init__(lam)

        self.M = M
        self.b = b
