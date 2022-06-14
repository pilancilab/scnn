"""Interfaces for external optimization routines / solvers.

TODO:
    - Extend LinearSolver to support vector-target problems by simultaneously solving each regression problem.
"""

from typing import Dict, Any, Tuple, Optional

import numpy as np

from scnn.private.models import Model, ConvexMLP
from scnn.private.utils.linear import iterative_solvers, preconditioners


class ExternalSolver:
    """Base class / interface for solvers that call external optimization
    routines (e.g. scipy.optimize).

    Subclasses should use '__call__' as an interface to the external routine.
    """

    def __call__(
        self, model: Model, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Call the external solver to fit a model."""

        raise NotImplementedError(
            "'ExternalSolver' subclasses must implement '__call__' as an interface to the external solver."
        )


class LinearSolver(ExternalSolver):

    """Interface to scipy.sparse.linalg' solvers for least-squares problems."""

    def __init__(
        self,
        method_name: str,
        max_iters: int,
        tol: float,
        preconditioner: Optional[str] = None,
    ):

        self.method_name = method_name
        self.max_iters = max_iters
        self.tol = tol
        self.preconditioner = preconditioner

    def __call__(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Model, Dict[str, Any]]:

        y = y.squeeze()
        if len(y.shape) > 1:
            raise ValueError(
                "LinearSolver only supports scalar output problems at the moment."
            )

        lam: float = 0.0
        if model.regularizer is not None:
            lam = model.regularizer.lam * len(y)

        M = None
        D = None
        if isinstance(model, ConvexMLP):
            D = model.D

        if self.preconditioner is not None:
            M = preconditioners.get_preconditioner(self.preconditioner, X, D)

        X_op = model.data_operator(X)

        optimal_weights, exit_status = iterative_solvers.lstsq_iterative_solve(
            X_op, y, lam, M, self.method_name, self.max_iters, self.tol
        )

        # reshape weights if necessary
        model.weights = optimal_weights.reshape(model.weights.shape)

        return model, exit_status
