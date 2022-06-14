"""Proximal gradient descent optimizer."""

from scnn.private.methods.core import (
    proximal_gradient_step,
    proximal_gradient_ls,
)
from scnn.private.methods.optimizers.proximal_optimizer import (
    ProximalOptimizer,
    ProximalLSOptimizer,
)
from scnn.private.methods.line_search import (
    StepSizeUpdater,
    LSCondition,
    Backtracker,
)
from scnn.private.prox import ProximalOperator


class PGD(ProximalOptimizer):

    """Proximal gradient descent (PGD) optimizer."""

    def __init__(
        self,
        step_size: float,
        prox: ProximalOperator,
        update_step_size: StepSizeUpdater,
    ):
        """
        :param step_size: the constant step-size to use.
        :param prox: a proximal operator. See 'proximal_ops'.
        :param update_step_size: a rule for updating the step-size after each line-search.
        """
        super().__init__(proximal_gradient_step, step_size, prox, update_step_size)


class PGDLS(ProximalLSOptimizer):
    """Proximal gradient descent with line-search."""

    def __init__(
        self,
        init_step_size: float,
        ls_cond: LSCondition,
        backtrack_fn: Backtracker,
        update_step_size: StepSizeUpdater,
        prox: ProximalOperator,
    ):
        """
        :param init_step_size: first step-size to try when running the line-search.
        :param ls_cond: the line-search condition to check.
        :param backtrack: a rule for calculating the next step-size to try.
        :param update_step_size: a rule for updating the step-size after each line-search.
        :param prox: a proximal operator. See 'proximal_ops'.
        """
        super().__init__(
            proximal_gradient_ls,
            init_step_size,
            ls_cond,
            backtrack_fn,
            update_step_size,
            prox,
        )
