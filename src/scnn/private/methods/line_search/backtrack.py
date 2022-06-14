"""Rules for backtracking and updating step-sizes during a line-search."""
from typing import List, Optional

import lab


class Backtracker:

    """Base class for backtracking conditions."""

    def __call__(
        self,
        step_size: float,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Pick the next candidate step-size for a line-search.

        :param step_size: the step_size previously tried.
        :returns: a new step-size to try.
        """
        raise NotImplementedError("Backtracking methods must implement '__call__'!")


class MultiplicativeBacktracker(Backtracker):

    """Step-size selection by simple multiplicative backtracking."""

    def __init__(self, beta=0.9):
        """
        :param beta: the backtracking parameter.
        """
        self.beta = beta

    def __call__(
        self,
        step_size: float,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Step-size selection by simple backtracking,

            step_size <- step_size * beta.
        :param step_size: the step_size previously tried.
        :returns: a new step-size to try.
        """

        return step_size * self.beta
