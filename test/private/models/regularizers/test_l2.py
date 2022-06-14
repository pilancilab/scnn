"""
Tests for l2 regularization.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from scnn.private.models import LinearRegression
from scnn.private.models.regularizers.l2 import L2Regularizer
from scnn.private.utils.linear import direct_solvers
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestL2Regularizer(unittest.TestCase):
    """Tests for L2 regularizers."""

    d: int = 5
    n: int = 10
    lam: float = 2.0
    rng: np.random.Generator = np.random.default_rng(778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)
        # generate dataset
        train_set, _, _ = gen_regression_data(self.rng, self.n, 0, self.d)
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)

        # initialize model
        self.regularizer = L2Regularizer(lam=self.lam)
        self.lr = LinearRegression(self.d)
        self.regularized_model = LinearRegression(
            self.d, regularizer=self.regularizer
        )
        self.objective, self.grad = self.regularized_model.get_closures(
            self.X, self.y
        )

        def objective_fn(v):
            return self.objective(lab.tensor(v))

        def grad_fn(v):
            return lab.to_np(self.grad(lab.tensor(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

        random_weights = lab.tensor(
            self.rng.standard_normal(self.d, dtype=self.dtype)
        )
        self.regularized_model.weights = random_weights

    def test_objective(self):
        """Check that the regularized objective is computed properly"""
        random_weights = self.regularized_model.weights

        regularized_loss = (
            lab.sum(((self.X @ random_weights) - self.y) ** 2)
        ) / (2 * len(self.y)) + (self.lam / 2) * lab.sum(random_weights ** 2)

        self.assertTrue(
            lab.allclose(
                regularized_loss,
                self.regularized_model.objective(self.X, self.y),
            ),
            "The regularized model objective did not match direct computation!",
        )

    def test_grad(self):
        """Check that the gradient is computed properly."""
        # test the gradient against brute computation
        random_weights = self.regularized_model.weights

        model_grad = self.regularized_model.grad(self.X, self.y)
        manual_grad = (
            self.X.T @ ((self.X @ random_weights) - self.y) / len(self.y)
            + self.lam * random_weights
        )

        self.assertTrue(
            lab.allclose(model_grad, manual_grad),
            "Gradient does not match brute computation.",
        )

        # check that the gradient is zero after the normal equations.
        wopt = direct_solvers.solve_ne(
            self.X, self.y, lam=self.lam * len(self.y)
        )
        grad = self.regularized_model.grad(self.X, self.y, wopt)
        self.assertTrue(
            lab.allclose(grad, lab.zeros_like(grad)),
            "The gradient should be zero after the normal equations.",
        )

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
