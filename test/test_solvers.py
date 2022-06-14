"""End-to-end tests for solvers.

TODO:
    - Test cone decomposition method once implemented.
"""

import unittest

import numpy as np

from scnn.private.utils.data.synthetic import gen_regression_data
from scnn.regularizers import NeuronGL1, FeatureGL1, L2
from scnn.solvers import RFISTA, AL, LeastSquaresSolver, CVXPYSolver
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors
from scnn.models import ConvexGatedReLU, ConvexReLU
from scnn.optimize import optimize_model


class TestRegularizers(unittest.TestCase):
    """Test solving convex reformulations with different regularizers."""

    def setUp(self):
        # Generate realizable synthetic classification problem (ie. Figure 1)
        n_train = 100
        n_test = 100
        d = 10
        self.c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(
            123, n_train, n_test, d, self.c, kappa=kappa
        )
        self.lam = 0.0001

        # Instantiate convex model and other options.
        self.G = sample_gate_vectors(np.random.default_rng(123), d, self.max_neurons)
        self.metrics = Metrics()
        self.regularizer = NeuronGL1(0.01)

    def test_RFISTA(self):
        """Test the R-FISTA solver on a gated ReLU problem."""
        model = ConvexGatedReLU(self.G, c=self.c)
        solver = RFISTA(model, tol=1e-6)
        model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

    def test_AL(self):
        """Test the augmented Lagrangian solver on a ReLU problem."""
        model = ConvexReLU(self.G, c=self.c)

        # terminate early.
        solver = AL(model, tol=1e-3, constraint_tol=1e-3)
        model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

    def test_least_squares(self):
        """Test the least-squares solver on an l2-regularized gated ReLU problem."""
        model = ConvexGatedReLU(self.G, c=1)
        solver = LeastSquaresSolver(model)
        model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train[:, 0],  # only supports scalar output for now.
            self.X_test,
            self.y_test,
            L2(0.01),
        )

    def test_CVXPY(self):
        """Test CVXPY interior-point solvers on Gated ReLU and ReLU problems."""
        model = ConvexGatedReLU(self.G, c=self.c)
        solver = CVXPYSolver(model, solver="ecos")
        model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

        model = ConvexReLU(self.G, c=self.c)
        solver = CVXPYSolver(model, solver="ecos")
        model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )


if __name__ == "__main__":
    unittest.main()
