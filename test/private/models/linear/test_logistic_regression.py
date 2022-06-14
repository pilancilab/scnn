"""
Tests for logistic regression.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from scnn.private.models import LogisticRegression
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestLogisticRegression(unittest.TestCase):
    """Tests for logistic regression."""

    d: int = 5
    n: int = 10
    rng: np.random.Generator = np.random.default_rng(seed=778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)
        # generate dataset
        train_set, _, self.wopt = gen_regression_data(self.rng, self.n, 0, self.d)
        self.X, self.y = lab.all_to_tensor(train_set)
        self.wopt = lab.tensor(self.wopt)
        self.y = lab.squeeze(self.y)
        self.y = lab.sign(self.y)

        # initialize model
        self.lr = LogisticRegression(self.d)
        self.objective, self.grad = self.lr.get_closures(self.X, self.y)

        def objective_fn(v):
            return self.objective(lab.tensor(v))

        def grad_fn(v):
            return lab.to_np(self.grad(lab.tensor(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_grad(self):
        """Check that the gradient is computed properly."""

        # check the gradient against finite differences
        for i in range(self.tries):
            v = self.rng.standard_normal(self.d, dtype=self.dtype)

            self.assertTrue(
                np.allclose(
                    check_grad(self.objective_fn, self.grad_fn, v),
                    0,
                    rtol=1e-5,
                    atol=1e-5,
                ),
                "Gradient did not match finite differences approximation.",
            )


if __name__ == "__main__":
    unittest.main()
