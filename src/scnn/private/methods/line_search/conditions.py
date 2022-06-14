"""Line search conditions and tools."""
from typing import List

import numpy as np

import lab

# classes


class LSCondition:

    """Base class for line-search conditions."""

    def __call__(
        self,
        f0: float,
        f1: float,
        step: lab.Tensor,
        grad: lab.Tensor,
        step_size: float,
    ) -> bool:
        raise NotImplementedError(
            "Line-search conditions must implement '__call__'!"
        )


class FSS:

    """Base class for line-search conditions."""

    def __call__(
        self,
        f0: float,
        f1: float,
        step: lab.Tensor,
        grad: lab.Tensor,
        step_size: float,
    ) -> bool:
        return True


class QuadraticBound(LSCondition):
    def __call__(
        self,
        f0: float,
        f1: float,
        step: lab.Tensor,
        grad: lab.Tensor,
        step_size: float,
    ) -> bool:
        """Check the Armijo or sufficient progress condition, which is.

            $f(x_{k+1} ≤ f(x_k) + <grad, step> + eta/2 ||step||_2^2$.
        :param f0: previous objective value, f(x_k).
        :param f1: new objective value, f(x_{k+1}).
        :param step: the (descent) step to check against the Armijo condition,
            ie. the difference between the new iterate and the previous one,
                $step = x_{k+1} - x_k$.
        :param grad: the grad at which to check the Armijo condition. Must have the same shape as 'step'.
        :param step_size: the step-size used to generate the step.
        :returns: boolean indicating whether or not the Armijo condition holds.
        """

        step = lab.ravel(step)
        grad = lab.ravel(grad)

        return 2.0 * step_size * (f1 - f0 - lab.dot(step, grad)) < lab.dot(
            step, step
        )


class DiagQB(LSCondition):

    """ """

    def __init__(self, A: lab.Tensor):
        """
        :param rho: the relaxation parameter for the linearization. A sensible default is 1e-4.
        """
        self.A = A

    def __call__(
        self,
        f0: float,
        f1: float,
        step: lab.Tensor,
        grad: lab.Tensor,
        step_size: float,
    ) -> bool:
        """Check the Armijo or sufficient progress condition, which is.

            $f(x_{k+1} ≤ f(x_k) + <grad, step> + eta/2 ||step||_2^2$.
        :param f0: previous objective value, f(x_k).
        :param f1: new objective value, f(x_{k+1}).
        :param step: the (descent) step to check against the Armijo condition,
            ie. the difference between the new iterate and the previous one,
                $step = x_{k+1} - x_k$.
        :param grad: the grad at which to check the Armijo condition. Must have the same shape as 'step'.
        :param step_size: the step-size used to generate the step.
        :returns: boolean indicating whether or not the Armijo condition holds.
        """

        step = lab.ravel(step)
        grad = lab.ravel(grad)

        return 2.0 * step_size * (f1 - f0 - lab.dot(step, grad)) < lab.dot(
            step, self.A * step
        )


class Armijo(LSCondition):

    """The Armijo line-search condition."""

    def __init__(self, rho: float = 1e-4):
        """
        :param rho: the relaxation parameter for the linearization. A sensible default is 1e-4.
        """
        self.rho = rho

    def __call__(
        self,
        f0: float,
        f1: float,
        step: lab.Tensor,
        grad: lab.Tensor,
        step_size: float,
    ) -> bool:
        """Check the Armijo or sufficient progress condition, which is.

            $f(x_{k+1} ≤ f(x_k) + rho <grad, step>$.
        :param f0: previous objective value, f(x_k).
        :param f1: new objective value, f(x_{k+1}).
        :param step: the (descent) step to check against the Armijo condition,
            ie. the difference between the new iterate and the previous one,
                $step = x_{k+1} - x_k$.
        :param grad: the grad at which to check the Armijo condition. Must have the same shape as 'step'.
        :param step_size: the step-size used to generate the step.
        :returns: boolean indicating whether or not the Armijo condition holds.
        """

        return f1 <= f0 + self.rho * lab.sum(lab.multiply(step, grad))
