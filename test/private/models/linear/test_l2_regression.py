"""
Tests for least-squares regression.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from scnn.private.models import LinearRegression
from scnn.private.utils.linear import direct_solvers
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestLinearRegression(unittest.TestCase):
    """Tests for least squares regression."""

    d: int = 5
    n: int = 10
    rng: np.random.Generator = np.random.default_rng(778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)
        # generate dataset
        train_set, _, self.wopt = gen_regression_data(self.rng, self.n, 0, self.d)
        self.X, self.y = lab.all_to_tensor(train_set)
        self.wopt = lab.tensor(self.wopt)
        self.y = lab.squeeze(self.y)

        # initialize model
        self.lr = LinearRegression(self.d)

        self.objective, self.grad = self.lr.get_closures(self.X, self.y)

        def objective_fn(v):
            return self.objective(lab.tensor(v))

        def grad_fn(v):
            return lab.to_np(self.grad(lab.tensor(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_grad(self):
        """Check that the gradient is computed properly."""
        # test the gradient against brute computation
        random_weights = lab.tensor(self.rng.standard_normal(self.d, dtype=self.dtype))
        self.lr.weights = random_weights

        model_grad = self.lr.grad(self.X, self.y)
        manual_grad = (self.X.T @ ((self.X @ random_weights) - self.y)) / len(self.y)

        self.assertTrue(
            np.allclose(model_grad, manual_grad),
            "Gradient does not match brute computation.",
        )

        # check that the gradient is zero after the normal equations.
        wopt = direct_solvers.solve_ne(self.X, self.y, lam=0)
        grad = self.lr.grad(self.X, self.y, wopt)
        self.assertTrue(lab.allclose(grad, lab.zeros_like(grad)))

        # test the gradient against finite differences
        for i in range(self.tries):
            v = self.rng.standard_normal(self.d, dtype=self.dtype)

            self.assertTrue(
                np.isclose(
                    check_grad(self.objective_fn, self.grad_fn, v),
                    0,
                    rtol=1e-5,
                    atol=1e-5,
                ),
                "Gradient does not match finite differences approximation.",
            )


if __name__ == "__main__":
    unittest.main()
