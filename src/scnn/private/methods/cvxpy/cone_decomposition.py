"""CVXPY solvers for the cone decomposition problem."""

from typing import Dict, Any, Tuple

from logging import root, INFO
import numpy as np
import cvxpy as cp

import lab

from scnn.private.models import Model, ConvexMLP, AL_MLP

from .cvxpy_solver import CVXPYSolver


class DecompositionProgram(CVXPYSolver):
    """Convert a Gated ReLU model into a ReLU model by cone decomposition.

    Base class for CVXPY programs that convert a convex Gated ReLU model into a
    convex ReLU model by decomposing the weights onto the constraint cones
    :math:`K_i` and :math:`-K_i`.
    """

    def get_objective(self, V, U) -> cp.Minimize:
        raise NotImplementedError(
            "All decomposition programs must implement an objective."
        )

    def __call__(
        self, model: Model, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Call the CVXPY solver to compute the decomposition."""

        assert isinstance(model, ConvexMLP)

        # lookup problem dimensions
        n, d = X.shape
        c = y.shape[1]
        p = model.D.shape[1]

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        D_np = lab.to_np(model.D)
        W = lab.to_np(model.weights.reshape(p * c, d))

        # create optimization variables
        V = cp.Variable((p * c, d))
        U = W + V

        # minimize two-norm of decompositions
        objective = self.get_objective(V, U)

        # constraints
        A = 2 * D_np - np.ones_like(D_np)

        constraints = [
            cp.multiply(A, X_np @ U[i * p : (i + 1) * p].T) >= 0
            for i in range(c)
        ]

        constraints += [
            cp.multiply(A, X_np @ V[i * p : (i + 1) * p].T) >= 0
            for i in range(c)
        ]

        problem = cp.Problem(objective, constraints)

        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(
            solver=self.solver, verbose=verbose, **self.solver_kwargs
        )

        # extract solution
        decomp_weights = lab.stack(
            [
                lab.tensor(U.value, dtype=lab.get_dtype()).reshape(c, p, d),
                lab.tensor(V.value, dtype=lab.get_dtype()).reshape(c, p, d),
            ]
        )
        relu_model = AL_MLP(
            d,
            model.D,
            model.U,
            model.kernel,
            regularizer=model.regularizer,
            c=model.c,
        )
        relu_model.weights = decomp_weights

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return relu_model, exit_status


class MinL2Decomposition(DecompositionProgram):
    """Decompose by minimize the sum of L2 norms."""

    def get_objective(self, V, U) -> cp.Minimize:
        return cp.Minimize(
            cp.sum(cp.pnorm(U, p=2, axis=1) + cp.pnorm(V, p=2, axis=1))
        )


class MinRelaxedL2Decomposition(DecompositionProgram):
    """Decompose by minimize the sum of L2 norms."""

    def get_objective(self, V, U) -> cp.Minimize:
        return cp.Minimize(cp.sum(V ** 2))


class MinL1Decomposition(DecompositionProgram):
    """Decompose by minimize the sum of L2 norms."""

    def get_objective(self, V, U) -> cp.Minimize:
        return cp.Minimize(
            cp.sum(cp.pnorm(U, p=1, axis=1) + cp.pnorm(V, p=1, axis=1))
        )


class FeasibleDecomposition(DecompositionProgram):
    """Decompose by minimize the sum of L2 norms."""

    def get_objective(self, V, U) -> cp.Minimize:
        return cp.Minimize(1)


class SOCPDecomposition(CVXPYSolver):
    """Base class for CVXPY programs that convert a convex gatedReLU model into
    a convex ReLU model by decomposing the weights onto the constraint cones
    :math:`K_i` and :math:`-K_i`."""

    def __call__(
        self, model: Model, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Call the CVXPY solver to compute the decomposition."""

        assert isinstance(model, ConvexMLP)
        if isinstance(model, (AL_MLP)):
            raise ValueError(
                "Cone decomposition is only defined for instances of 'ConvexMLP'"
            )

        # lookup problem dimensions
        n, d = X.shape
        c = y.shape[1]
        p = model.D.shape[1]

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        D_np = lab.to_np(model.D)
        W = lab.to_np(model.weights.reshape(p * c, d))

        # create optimization variables
        V = cp.Variable((p * c, d))
        U = W + V

        # linear components of the SOCP
        t = cp.Variable((p * c))
        l = cp.Variable((p * c))

        # linear objective
        objective = cp.Minimize(cp.sum(t + l))

        # SOCP constraints
        constraints = [cp.SOC(t, V, axis=1)]
        constraints += [cp.SOC(l, U, axis=1)]

        # cone constraints
        A = 2 * D_np - np.ones_like(D_np)

        constraints += [
            cp.multiply(A, X_np @ U[i * p : (i + 1) * p].T) >= 0
            for i in range(c)
        ]

        constraints += [
            cp.multiply(A, X_np @ V[i * p : (i + 1) * p].T) >= 0
            for i in range(c)
        ]

        problem = cp.Problem(objective, constraints)

        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(
            solver=self.solver, verbose=verbose, **self.solver_kwargs
        )

        # extract solution
        decomp_weights = lab.stack(
            [
                lab.tensor(U.value, dtype=lab.get_dtype()).reshape(c, p, d),
                lab.tensor(V.value, dtype=lab.get_dtype()).reshape(c, p, d),
            ]
        )
        relu_model = AL_MLP(
            d,
            model.D,
            model.U,
            model.kernel,
            regularizer=model.regularizer,
            c=model.c,
        )
        relu_model.weights = decomp_weights

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return relu_model, exit_status
