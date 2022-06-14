"""Solvers for convex reformulations of neural networks based on the `CVXPY
<https://www.cvxpy.org>`_ DSL.

TODO:
    - How to handle GL1 penalties for the convex LassoNet models?
"""

from typing import Dict, Any, Tuple, List, Optional

from logging import root, INFO
import numpy as np
import cvxpy as cp

import lab

from scnn.private.models import (
    Model,
    Regularizer,
    L2Regularizer,
    GroupL1Regularizer,
    FeatureGroupL1Regularizer,
    ConvexMLP,
    AL_MLP,
)

from .cvxpy_solver import CVXPYSolver


class ConvexReformulationSolver(CVXPYSolver):
    """Solver for convex reformulations of neural network based on `CVXPY
    <https://www.cvxpy.org>`_.

    This is an abstract class that wraps CVXPY expressions used in all of the
    convex reformulations.
    """

    c: int
    d: int
    p: int
    n: int

    def process_inputs(
        self, model: ConvexMLP, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast inputs into NumPy arrays and save problem dimensions."""

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        y_np = lab.to_np(y)
        D_np = lab.to_np(model.D)

        # lookup problem dimensions
        self.n, self.d = X_np.shape
        self.c = y_np.shape[1]
        self.p = D_np.shape[1]

        return X_np, y_np, D_np

    def get_cone_constraints(
        self, W: cp.Variable, X: np.ndarray, A: np.ndarray
    ) -> List[cp.Expression]:
        """Form polyhedral cone constraints for the convex reformulation of a
        ReLU neural network.

        Args:
            W: a :math:`p * c \\times d` matrix of variables upon which to place the constraints,
                where `p` is the number of activation patterns, c is the target dimension, and `d`
                is the input dimension.
            X: a :math:`n \\times d` matrix of training examples.
            A: a :math:`d \\times p` matrix of signed activation patterns: :math:`A = 2 D - I`.

        Returns:
            A list of :math:`p * c` constraints, one for each of the target-activation pair.
        """
        return [
            cp.multiply(A, X @ W[i * self.p : (i + 1) * self.p].T) >= 0
            for i in range(self.c)
        ]

    def get_squared_error(
        self, W: cp.Variable, X: np.ndarray, y: np.ndarray, D: np.ndarray
    ) -> cp.Expression:
        """Form CVXPY expression for the squared-error in the convex
        reformulation.

        Args:
            W: a :math:`p * c \\times d` matrix of variables for the model weights,
                where `p` is the number of activation patterns, c is the target dimension, and `d`
                is the input dimension.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c' matrix of targets.
            D: a :math:`d \\times p` matrix of activation patterns.

        Returns:
           An expression for the model's squared-error.
        """

        loss = 0.0
        for i in range(self.c):
            loss += (
                cp.sum_squares(
                    cp.sum(
                        cp.multiply(D, X @ W[i * self.p : (i + 1) * self.p].T),
                        axis=1,
                    )
                    - y[:, i]
                )
                / (2 * self.n * self.c)
            )

        return loss

    def get_regularization(
        self, W: cp.Variable, regularizer: Optional[Regularizer] = None
    ) -> cp.Expression:
        """Form CVXPY expression for the model regularization.

        Args:
            W: a :math:`p * c \\times d` matrix of variables for the model weights,
                where `p` is the number of activation patterns, c is the target dimension, and `d`
                is the input dimension.
            regularizer: the model regularizer.

        Returns:
           An expression for the regularization.
        """
        if regularizer is None or regularizer.lam == 0:
            return 0.0

        lam = regularizer.lam

        if isinstance(regularizer, L2Regularizer):
            return (lam / 2) * cp.sum_squares(W)
        elif isinstance(regularizer, GroupL1Regularizer):
            return lam * cp.mixed_norm(W, p=2, q=1)
        elif isinstance(regularizer, FeatureGroupL1Regularizer):
            return lam * cp.mixed_norm(W.T, p=2, q=1)
        else:
            raise ValueError(f"Regularizer {regularizer} not recognized!")


class CVXPYGatedReLUSolver(ConvexReformulationSolver):
    """CVXPY-based solver for convex Gated ReLU models."""

    def __call__(
        self, model: ConvexMLP, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Solve the convex reformulation for two-layer models with Gated ReLU
        activations.

        Args:
            model: the convex formulation to optimize.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c' matrix of targets.

        Returns:
           The convex reformulation with optimized weights.
        """

        X_np, y_np, D_np = self.process_inputs(model, X, y)

        # create optimization variables
        W = cp.Variable((self.p * self.c, self.d))

        # get squared-error
        loss = self.get_squared_error(W, X_np, y_np, D_np)
        loss += self.get_regularization(W, model.regularizer)

        objective = cp.Minimize(loss)

        problem = cp.Problem(objective)

        # infer verbosity of sub-solver.
        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(
            solver=self.solver, verbose=verbose, **self.solver_kwargs
        )

        # extract solution
        model.weights = lab.tensor(W.value, dtype=lab.get_dtype()).reshape(
            self.c, self.p, self.d
        )

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return model, exit_status


class CVXPYReLUSolver(ConvexReformulationSolver):
    """CVXPY-based solver for convex ReLU models."""

    def __call__(
        self, model: AL_MLP, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Solve the convex reformulation for two-layer models with ReLU
        activations.

        Args:
            model: the convex formulation to optimize.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c' matrix of targets.

        Returns:
           The convex reformulation with optimized weights.
        """
        X_np, y_np, D_np = self.process_inputs(model, X, y)

        # create optimization variables
        U = cp.Variable((self.p * self.c, self.d))
        V = cp.Variable((self.p * self.c, self.d))
        W = U - V

        # get squared-error
        loss = self.get_squared_error(W, X_np, y_np, D_np)
        loss += self.get_regularization(U, model.regularizer)
        loss += self.get_regularization(V, model.regularizer)

        objective = cp.Minimize(loss)

        # define orthant constraints
        A = 2 * D_np - np.ones_like(D_np)
        constraints = self.get_cone_constraints(
            U, X_np, A
        ) + self.get_cone_constraints(V, X_np, A)

        problem = cp.Problem(objective, constraints)

        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(
            solver=self.solver, verbose=verbose, **self.solver_kwargs
        )

        # extract solution
        model.weights = lab.stack(
            [
                lab.tensor(U.value, dtype=lab.get_dtype()).reshape(
                    self.c, self.p, self.d
                ),
                lab.tensor(V.value, dtype=lab.get_dtype()).reshape(
                    self.c, self.p, self.d
                ),
            ]
        )

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return model, exit_status
