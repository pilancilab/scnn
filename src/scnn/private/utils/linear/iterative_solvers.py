"""Iterative methods for solving linear systems and least-squares problems."""
from typing import Optional, Dict, Any, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr, cg  # type: ignore


# methods

LSQR = "lsqr"
LSMR = "lsmr"
CG = "cg"

SOLVERS = [LSQR, LSMR, CG]

# parameters

TOL = 1e-6

# least-squares solvers.


def lstsq_iterative_solve(
    linear_op: LinearOperator,
    targets: np.ndarray,
    lam: float = 0.0,
    preconditioner: Optional[LinearOperator] = None,
    solver: str = LSMR,
    max_iters: int = 1000,
    tol: float = TOL,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Use an iterative method to solve the l2-regularized least-squares
    problem,

    .. math::

        \|X w - y\|_2^2 + (\lambda/2) * \|w\|_2^2,

    where X is given by 'linear_op', y by 'targets', and lambda by 'lam'.
    WARNING: this function *only* supports NumPy and cannot be used with the generic backend
    in 'scnn.private.backend'.
    :param linear_op: a generalized linear-operator that evaluates matrix-vector products for
    the data matrix matrix X.
    :param targets: the response/targets y to predict.
    :param lam: (optional) the strength of the l2 regularization used. Defaults to no regularization.
    See [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) for details.
    :param preconditioner: (optional) a preconditioner to use when solving the system.
    See 'methods/linear/preconditioners.py'.
    :param solver: (optional) which solver to use. Valid options are LMSR and LSQR.
    :param max_iters: (optional) the maximum number of iterations to run the solver.
    :param tol: (optional) the tolerance to which the least-squares problem should be solved.
    :returns: (solution, exit_status) -- the solution to the least-squares problem and status of the solver.
    """
    assert isinstance(targets, np.ndarray)  # only support NumPy arrays.

    damping = 0.0
    if lam != 0.0:
        damping = np.sqrt(lam)

    if preconditioner is not None:
        # compose the operators.
        linear_op = linear_op.dot(preconditioner)

    if solver == LSMR:
        w_opt, istop, itn = lsmr(
            A=linear_op,
            b=targets,
            damp=damping,
            maxiter=max_iters,
            atol=tol,
            btol=tol,
        )[0:3]

        success = istop != 7
    elif solver == LSQR:
        w_opt, istop, itn, = lsqr(
            A=linear_op,
            b=targets,
            damp=damping,
            iter_lim=max_iters,
            atol=tol,
            btol=tol,
        )[0:3]

        success = istop != 7
    else:
        raise ValueError(f"Iterative solver {solver} not recognized!")

    # undo affect of preconditioning.
    if preconditioner is not None:
        w_opt = preconditioner.dot(w_opt)

    exit_status = {"success": success, "stop_condition": istop, "iterations": itn}

    return w_opt, exit_status


def linear_iterative_solve(
    linear_op: LinearOperator,
    targets: np.ndarray,
    preconditioner: Optional[LinearOperator] = None,
    solver: str = CG,
    max_iters: int = 1000,
    tol: float = TOL,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Use an iterative method to solve the linear system.

    .. math::

        Xw = b,

    where X is given by 'linear_op', y by 'targets', and lambda by 'lam'.
    WARNING: this function *only* supports NumPy and cannot be used with the generic backend
    in 'scnn.private.backend'.
    :param linear_op: a generalized linear-operator that evaluates matrix-vector products for
    the data matrix matrix X.
    :param targets: the response/targets y to predict.
    :param preconditioner: (optional) a preconditioner to use when solving the system.
    See 'methods/linear/preconditioners.py'.
    :param solver: (optional) which solver to use. Valid options are CG.
    :param max_iters: (optional) the maximum number of iterations to run the solver.
    :returns: (solution, exit_status) -- the solution to the least-squares problem and status of the solver.
    """
    assert isinstance(targets, np.ndarray)  # only support NumPy arrays.

    if linear_op.shape[0] != linear_op.shape[1]:
        raise ValueError(
            "'linear_op' must specify a square matrix. Use 'lstsq_iterative_solve' for rectangular systems."
        )

    # flatten while solving the system.
    flat_targets = targets.reshape(-1)

    if solver == CG:
        w_opt, info = cg(
            A=linear_op,
            b=flat_targets,
            maxiter=max_iters,
            M=preconditioner,
            tol=tol,
            atol="legacy",
        )
        success = info == 0
    else:
        raise ValueError(f"Iterative solver {solver} not recognized!")

    exit_status = {"success": success}

    # restore original shape.
    w_opt = w_opt.reshape(targets.shape)

    return w_opt, exit_status
