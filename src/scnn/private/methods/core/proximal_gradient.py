"""Proximal gradient descent with line-search."""

from typing import Callable, Tuple, Dict, Any
from functools import partial

import numpy as np

import lab

from scnn.private.methods.core import ls
from scnn.private.methods.line_search import Backtracker, LSCondition
from scnn.private.prox import ProximalOperator

# step operators


def proximal_gradient_step(
    w: lab.Tensor,
    descent_dir: lab.Tensor,
    step_size: float,
    prox: ProximalOperator,
) -> lab.Tensor:
    """Take one step of proximal gradient descent.

    :param w: the parameters to be updated.
    :param grad: the gradient of the smooth component of the loss function.
    :param step_size: the step-size to use.
    :param prox: the proximal operator. It must take a step-size parameter.
    :returns: updated parameters
    """

    w_plus = w - step_size * descent_dir
    return prox(w_plus, step_size)


def fista_step(
    w: lab.Tensor,
    descent_dir: lab.Tensor,
    step_size: float,
    prox: ProximalOperator,
    v: lab.Tensor,
    t: float,
) -> Tuple[lab.Tensor, lab.Tensor, float]:
    """
    :param w: the parameters to be updated.
    :param grad: the gradient of the smooth component of the loss function.
    :param step_size: the step-size to use.
    :param prox: the proximal operator. It must take a step-size parameter.
    :param v: the extrapolation sequence.
    :param t: the extrapolation parameter.
    :returns: (w_plus, v_plus, t_plus) --- the updated sequences.
    """

    w_plus = proximal_gradient_step(v, descent_dir, step_size, prox)
    t_plus = 1 + np.sqrt(1 + 4 * t ** 2) / 2
    v_plus = w_plus + (t - 1) * (w_plus - w) / t_plus

    return w_plus, v_plus, t_plus


# main method: proximal gradient descent with line-search.


def proximal_gradient_ls(
    w: lab.Tensor,
    f0: float,
    descent_dir: lab.Tensor,
    grad: lab.Tensor,
    obj_fn: Callable,
    grad_fn: Callable,
    init_step_size: float,
    ls_cond: LSCondition,
    backtrack: Backtracker,
    prox: ProximalOperator,
) -> Tuple[lab.Tensor, float, float, Dict[str, Any]]:
    """Take one step of proximal gradient descent using a line-search to pick
    the step-size.

    :param w: the parameters to be updated.
    :param f0: the objective function evaluated at w.
    :param descent_dir: the descent direction for the step.
    :param grad: the gradient of the smooth component of the loss function.
    :param obj_fn: function that returns the objective when called at w.
    :param grad_fn: function that returns the gradient of the smooth component of the loss function.
    :param init_step_size: the first step-size to use.
    :param ls_cond: the line-search condition to check.
    :param backtrack: a rule for calculating the next step-size to try.
    :param prox: the proximal operator. It must take a step-size parameter.
    :returns: (w_next, f1, step_size, exit_state): the updated parameters, objective value, new step-size, and exit state.
    """

    # input the proximal operator and extra arguments.
    test_fn = partial(proximal_gradient_step, prox=prox)

    # line-search *through* the proximal operator.
    return ls(
        test_fn,
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


def fista_ls(
    w: lab.Tensor,
    f0: float,
    descent_dir: lab.Tensor,
    grad: lab.Tensor,
    obj_fn: Callable,
    grad_fn: Callable,
    init_step_size: float,
    ls_cond: LSCondition,
    backtrack: Backtracker,
    prox: ProximalOperator,
    v: lab.Tensor,
    t: float,
    mu: float = 0.0,
) -> Tuple[lab.Tensor, lab.Tensor, float, float, float, Dict[str, Any]]:
    """Take one step of proximal gradient descent using a line-search to pick
    the step-size.

    :param w: the parameters to be updated.
    :param f0: the objective function evaluated at w.
    :param descent_dir: the descent direction for the step.
    :param grad: the gradient of the smooth component of the loss function.
    :param obj_fn: function that returns the objective when called at w.
    :param grad_fn: function that returns the gradient of the smooth component of the loss function.
    :param init_step_size: the first step-size to use.
    :param ls_cond: the line-search condition to check.
    :param backtrack: a rule for calculating the next step-size to try.
    :param prox: the proximal operator. It must take a step-size parameter.
    :param v: the extrapolation sequence.
    :param t: the extrapolation parameter.
    :param mu: (optional) a lower-bound on the strong-convexity parameter of the objective.
        The method defaults to the parameter sequence for non-strongly convex functions when
        mu is not supplied.
    :returns: (w_next, f1, step_size, exit_state): the updated parameters, objective value, new step-size, and exit state.
    """

    # line-search starting from secondary (extrapolation) sequence.
    w_plus, f1, step_size, exit_state = proximal_gradient_ls(
        v,
        f0,
        descent_dir,
        grad,
        obj_fn,
        grad_fn,
        init_step_size,
        ls_cond,
        backtrack,
        prox,
    )
    # release memory
    t_plus = t

    if mu == 0.0:
        t_plus = 1 + np.sqrt(1 + 4 * t ** 2) / 2
        beta = (t - 1) / t_plus
    else:
        sqrt_kappa = np.sqrt(1 / (step_size * mu))
        beta = (sqrt_kappa - 1) / (sqrt_kappa + 1)

    v = w_plus + beta * (w_plus - w)

    return w_plus, v, t_plus, f1, step_size, exit_state
