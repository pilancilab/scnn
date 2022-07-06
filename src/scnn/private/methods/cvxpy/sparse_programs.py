"""Training programs for feature-sparse methods based on CVXPY.
"""

from typing import Dict, Any, Tuple, List, Optional
from itertools import product

from logging import root, INFO
import numpy as np
import cvxpy as cp

import lab

from scnn.private.models import (
    Model,
    CardinalityConstraint,
    ConvexMLP,
    AL_MLP,
)

from .training_programs import ConvexReformulationSolver


class CVXPYExactSparsitySolver(ConvexReformulationSolver):
    """CVXPY-based solver for convex reformulations with exact feature sparsity.

    Note that the resulting problem is a mixed-integer program.
    """

    def get_cardinality_variables(self) -> Tuple[cp.Variable, List[cp.Expression]]:

        return cp.Variable((self.d), boolean=True), []

    def get_cardinality_constraints(
        self, W: cp.Variable, E: cp.Variable, M: float
    ) -> List[cp.Expression]:
        """Generate cardinality constraints."""

        return [
            cp.abs(W[i, j]) <= M * E[j]
            for i, j in product(range(self.p * self.c), range(self.d))
        ]

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
        # check and unpack regularization argument
        assert isinstance(model.regularizer, CardinalityConstraint)

        b: int = model.regularizer.b
        M: float = model.regularizer.M

        X_np, y_np, D_np = self.process_inputs(model, X, y)

        # create optimization variables
        U = cp.Variable((self.p * self.c, self.d))
        V = cp.Variable((self.p * self.c, self.d))
        E, E_constraints = self.get_cardinality_variables()
        W = U - V

        # get squared-error
        loss = self.get_squared_error(W, X_np, y_np, D_np)
        loss += self.get_regularization(U, model.regularizer)
        loss += self.get_regularization(V, model.regularizer)

        objective = cp.Minimize(loss)

        # define orthant constraints
        A = 2 * D_np - np.ones_like(D_np)
        constraints = self.get_cone_constraints(U, X_np, A) + self.get_cone_constraints(
            V, X_np, A
        )

        # create support constraints
        constraints += [cp.sum(E) <= b]
        constraints += self.get_cardinality_constraints(U, E, M)
        constraints += self.get_cardinality_constraints(V, E, M)
        constraints += E_constraints

        problem = cp.Problem(objective, constraints)

        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(solver=self.solver, verbose=verbose, **self.solver_kwargs)

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
            "support": E.value,
        }

        return model, exit_status


class CVXPYRelaxedSparsitySolver(ConvexReformulationSolver):
    """CVXPY-based solver for convex reformulations with exact feature sparsity.

    Note that the resulting problem is a mixed-integer program.
    """

    def get_cardinality_variables(self) -> Tuple[cp.Variable, List[cp.Expression]]:
        """Get relaxed version of the cardinality variables."""

        E = cp.Variable((self.d), nonneg=True)
        return E, [E <= 1]
