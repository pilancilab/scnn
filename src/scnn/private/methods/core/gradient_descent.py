"""Gradient descent with line-search."""
from typing import Callable, Tuple, Dict, Any, Optional

import lab

from scnn.private.methods.core.line_search import ls
from scnn.private.methods.line_search import Backtracker, LSCondition
from scnn.private.prox import ProximalOperator


def gradient_step(
    w: lab.Tensor,
    grad: lab.Tensor,
    step_size: float,
    prox: Optional[ProximalOperator] = None,
) -> lab.Tensor:
    """Take one step of gradient descent.

    :param w: the parameters to be updated.
    :param grad: the gradient of the objective function.
    :param step_size: the step-size to use.
    :param prox: NOT USED. Included to unify step signatures.
    :returns: updated parameters
    """

    return w - step_size * grad


def gd_ls(
    w: lab.Tensor,
    f0: float,
    descent_dir: lab.Tensor,
    grad: lab.Tensor,
    obj_fn: Callable,
    grad_fn: Callable,
    init_step_size: float,
    ls_cond: LSCondition,
    backtrack: Backtracker,
    prox: Optional[ProximalOperator] = None,
) -> Tuple[lab.Tensor, float, float, Dict[str, Any]]:
    """Take one step of gradient descent using a line-search to pick the step-
    size.

    :param w: the parameters to be updated.
    :param f0: the objective function evaluated at w.
    :param descent_dir: the descent direction for the step.
    :param grad: the gradient of the smooth component of the loss function.
    :param obj_fn: function that returns the objective when called at w.
    :param grad_fn: function that returns the gradient of the objective function.
    :param init_step_size: the first step-size to use.
    :param ls_cond: the line-search condition to check.
    :param backtrack: a rule for calculating the next step-size to try.
    :param prox: NOT USED. Included to unify line-search signatures.
    :returns: (w_next, f1, step_size, exit_state): the updated parameters, objective value, new step-size, and exit state.
    """

    return ls(
        gradient_step,
        w,
        f0,
        descent_dir,
        grad,
        obj_fn,
        grad_fn,
        init_step_size,
        ls_cond,
        backtrack,
    )
