"""Rules for update step-sizes after a line-search."""
from typing import Optional

import lab


class StepSizeUpdater:

    """Base class for methods to update step-sizes between iterations."""

    def __call__(
        self,
        new_step_size: float,
        old_step_size: float,
        s: Optional[lab.Tensor] = None,
        f0: Optional[float] = None,
        f1: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Compute a new step-size given the result of the most recent line-
        search and the step-size from the previous iteration.

        :param new_step_size: the step-size most recently returned by the line-search.
        :param old_step-size: the step-size from the previous iteration.
        :param s: the difference in iterates: w_k - w_{k-1}
        """

        raise NotImplementedError("A step-size update must implement '__call__'!")


class KeepNew(StepSizeUpdater):

    """Simple update rule which returns step-size from the most recent line-
    search."""

    def __call__(
        self,
        new_step_size: float,
        old_step_size: float,
        s: Optional[lab.Tensor] = None,
        f0: Optional[float] = None,
        f1: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Compute a new step-size given the result of the most recent line-
        search and the step-size from the previous iteration.

        :param new_step_size: the step-size most recently returned by the line-search.
        :param old_step-size: the step-size from the previous iteration.
        :param s: the difference in iterates: w_k - w_{k-1}
        """

        return new_step_size


class KeepOld(StepSizeUpdater):

    """Simple update rule which returns the old step-size."""

    def __call__(
        self,
        new_step_size: float,
        old_step_size: float,
        s: Optional[lab.Tensor] = None,
        f0: Optional[float] = None,
        f1: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Compute a new step-size given the result of the most recent line-
        search and the step-size from the previous iteration.

        :param new_step_size: the step-size most recently returned by the line-search.
        :param old_step-size: the step-size from the previous iteration.
        :param s: the difference in iterates: w_k - w_{k-1}
        """

        return old_step_size


class ForwardTrack(StepSizeUpdater):

    """Compute the new step-size by slightly increasing the value returned by
    the line-search."""

    def __init__(self, alpha: float = 1.1):
        """
        :param alpha: the amount to multiplicatively increase the step-size by at each iteration.
        """
        self.alpha = alpha

    def __call__(
        self,
        new_step_size: float,
        old_step_size: float,
        s: Optional[lab.Tensor] = None,
        f0: Optional[float] = None,
        f1: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Compute a new step-size given the result of the most recent line-
        search and the step-size from the previous iteration.

        :param new_step_size: the step-size most recently returned by the line-search.
        :param old_step-size: the step-size from the previous iteration.
        :param s: the difference in iterates: w_k - w_{k-1}
        """

        return new_step_size * self.alpha


class Lassplore(StepSizeUpdater):

    """Step-size update rule proposed in 'Large Scale Spare Logistic
    Regression' by Liu et al.

    [https://dl.acm.org/doi/abs/10.1145/1557019.1557082]
    """

    def __init__(self, alpha: float = 1.2, threshold: float = 5.0):
        """
        :param alpha: the amount to multiplicatively increase the step-size by at each iteration.
        :param threshold: threshold for increasing the step-size after a successful line-search.
        """
        self.alpha = alpha
        self.threshold = threshold

    def __call__(
        self,
        new_step_size: float,
        old_step_size: float,
        s: Optional[lab.Tensor] = None,
        f0: Optional[float] = None,
        f1: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> float:
        """Compute a new step-size given the result of the most recent line-
        search and the step-size from the previous iteration.

        :param new_step_size: the step-size most recently returned by the line-search.
        :param old_step-size: the step-size from the previous iteration.
        :param s: the difference in iterates: w_k - w_{k-1}
        :param grad: the current gradient g_k.
        """
        # required arguments
        assert f0 is not None
        assert f1 is not None
        assert grad is not None
        assert s is not None

        denom = (f1 - (f0 + lab.sum(lab.multiply(grad, s)))) * 2 * new_step_size
        gap = lab.sum(s ** 2)

        if gap > self.threshold * denom:
            return new_step_size * self.alpha
        else:
            return new_step_size
