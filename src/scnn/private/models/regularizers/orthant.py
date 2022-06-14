"""Orthant constraint."""
from typing import Optional

import lab

from scnn.private.models.regularizers.constraint import Constraint

from scnn.private.prox import Orthant


class OrthantConstraint(Constraint):

    """Representation of the orthant constraint, A_i x >= 0, where A_i is a
    diagonal matrix with (A_i)_jk in {-1, 1}, as a regularizer.

    In essence, this is a wrapper for computation of the gradient mapping.
    """

    lam = 0.0

    def __init__(self, A: lab.Tensor):
        """
        :param A: a matrix of sign patterns defining orthants on which to project.
            The diagonal A_i is stored as the i'th column of A.
        :param lam: the tuning parameter controlling the strength of regularization.
        """

        self.projection_op = Orthant(A)
