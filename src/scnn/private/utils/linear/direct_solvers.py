"""Solvers for linear problems."""

import lab


def solve_ne(X: lab.Tensor, y: lab.Tensor, lam: float = 0.0) -> lab.Tensor:
    """Fit a linear model by directly solving the normal equations. This
    implementation forms X.T X and then solves the linear system.

    .. math::

        X^\top X w = X^\top b

    It is not memory efficient and should not be used when performance
    is required.
    :param X: (n,d) array containing the data examples.
    :param y: (n,d) array containing the data targets.
    :param lam: (optional) regularization parameter.
    :returns: the fit model.
    """
    n, d = X.shape

    # form X' X and X' y:
    XtX = X.T @ X
    Xty = X.T @ y

    return lab.solve(XtX + lam * lab.eye(d), Xty)
