"""Basic line-search loop."""

from typing import Callable, Dict, Tuple, Any

import lab
from scnn.private.methods.line_search import Backtracker, LSCondition

# constants

MAX_ATTEMPTS: int = 50
FAILURE_STEP: float = 1e-8


# line-search


def ls(
    test_fn: Callable,
    w: lab.Tensor,
    f0: float,
    descent_dir: lab.Tensor,
    grad: lab.Tensor,
    obj_fn: Callable,
    grad_fn: Callable,
    init_step_size: float,
    ls_cond: LSCondition,
    backtrack: Backtracker,
) -> Tuple[lab.Tensor, float, float, Dict[str, Any]]:
    """Take one step of a iterative method using a line-search to pick the
    step-size.

    :param test_fn: function that returns the next test point given w, grad, and the next step_size.
    :param w: the parameters to be updated.
    :param f0: the objective function evaluated at w.
    :param descent_dir: the descent direction for the step.
    :param grad: the gradient of the smooth component of the loss function.
    :param obj_fn: function that returns the objective when called at w.
    :param grad_fn: function that returns the gradient of the smooth component of the loss function.
    :param init_step_size: the first step-size to use.
    :param ls_cond: the line-search condition to check.
    :param backtrack: a rule for calculating the next step-size to try.
    :returns: (w_next, f1, step_size, exit_state) -- the updated parameters, objective value, new step-size, and exit state.
    """
    # setup
    success = True
    step_size = init_step_size

    # run line-search
    w_plus = test_fn(w, descent_dir, step_size)
    f1 = obj_fn(w_plus)
    attempts = 1

    while not ls_cond(f0, f1, w_plus - w, grad, step_size):
        step_size = backtrack(step_size)
        w_plus = test_fn(w, descent_dir, step_size)
        f1 = obj_fn(w_plus)
        attempts += 1

        # exceeded max attempts
        if attempts > MAX_ATTEMPTS:
            step_size = FAILURE_STEP
            w_plus = test_fn(w, grad, step_size)
            success = False
            break

    exit_state = {"attempts": attempts, "success": success, "step_size": step_size}

    return w_plus, f1, step_size, exit_state
