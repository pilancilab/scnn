"""Gradient descent optimizer."""

from scnn.private.methods.core import gradient_step, gd_ls
from scnn.private.methods.optimizers.optimizer import Optimizer, LSOptimizer
from scnn.private.methods.line_search import StepSizeUpdater, LSCondition, Backtracker


class GD(Optimizer):

    """Simple (sub)gradient descent optimizer."""

    def __init__(
        self,
        step_size: float,
        update_step_size: StepSizeUpdater,
    ):
        """
        :param step_size: the constant step-size to use.
        :param update_step_size: a rule for updating the step-size after each line-search.
        """
        super().__init__(gradient_step, step_size, update_step_size)


class GDLS(LSOptimizer):

    """Gradient descent with line-search."""

    def __init__(
        self,
        init_step_size: float,
        ls_cond: LSCondition,
        backtrack_fn: Backtracker,
        update_step_size: StepSizeUpdater,
    ):
        """
        :param init_step_size: first step-size to try when running the line-search.
        :param ls_cond: the line-search condition to check.
        :param backtrack: a rule for calculating the next step-size to try.
        :param update_step_size: a rule for updating the step-size after each line-search.
        """
        super().__init__(gd_ls, init_step_size, ls_cond, backtrack_fn, update_step_size)
