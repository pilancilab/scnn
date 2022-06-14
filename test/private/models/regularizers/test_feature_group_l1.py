"""
Tests for feature-wise group-l1 regularization.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from scnn import activations
from scnn.private.models import ConvexMLP
from scnn.private.models.convex import operators
from scnn.private.models import (
    FeatureGroupL1Regularizer,
)
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestFeatureGroupL1Regularizer(unittest.TestCase):
    """Tests for feature-wise group L1 regularizers."""

    d: int = 2
    n: int = 4
    c: int = 3
    lam: float = 2.0
    rng: np.random.Generator = np.random.default_rng(778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        # generate dataset
        train_set, _, self.wopt = gen_regression_data(
            self.rng, self.n, 0, self.d, c=self.c
        )
        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.wopt = lab.tensor(self.wopt)

        # initialize model
        self.p = self.D.shape[1]

        self.regularizer = FeatureGroupL1Regularizer(lam=self.lam)
        self.regularized_model = ConvexMLP(
            self.d, self.D, self.U, regularizer=self.regularizer, c=self.c
        )

        self.random_weights = self.rng.standard_normal(
            (self.c, self.p, self.d), dtype=self.dtype
        )
        self.regularized_model.weights = lab.tensor(self.random_weights * 100)

        self.objective, self.grad = self.regularized_model.get_closures(
            self.X, self.y
        )

        def objective_fn(v=None):
            if v is not None:
                v = lab.tensor(v).reshape(self.c, self.p, self.d)

            return self.objective(v)

        def grad_fn(v=None):
            if v is not None:
                v = lab.tensor(v).reshape(self.c, self.p, self.d)

            return lab.to_np(lab.ravel(self.grad(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_objective(self):
        """Check that the regularized objective is computed properly"""
        # the regularized model should have the same objective as the underlying model when lam = 0
        self.regularized_model.regularizer.lam = 0

        self.assertTrue(
            lab.allclose(
                self.regularized_model.objective(
                    self.X, self.y, ignore_regularizer=True
                ),
                self.regularized_model.objective(self.X, self.y),
            ),
            "The regularized model objective did not match the underlying model objective when lam = 0!",
        )

    def test_grad(self):
        """Check that the gradient is computed properly."""
        # the gradient should be the same as the underlying model gradient when lam = 0.
        self.regularized_model.regularizer.lam = 0
        model_grad = self.regularized_model.grad(self.X, self.y)

        self.assertTrue(
            lab.allclose(
                model_grad,
                self.regularized_model.grad(
                    self.X, self.y, ignore_regularizer=True
                ),
            ),
            "Gradient does not match underlying model gradient when lambda = 0.",
        )

        self.regularized_model.regularizer.lam = 1.0
        model_grad = self.regularized_model.grad(self.X, self.y)
        self.assertFalse(
            lab.allclose(
                model_grad,
                self.regularized_model.grad(
                    self.X, self.y, ignore_regularizer=True
                ),
            ),
            "Gradient matched underlying model gradient when lambda != 0.",
        )

        # set some weights to zero and check that their gradient is zero when lambda is large enough.
        self.regularized_model.regularizer.lam = 100000
        self.regularized_model.weights[:, :, 0] = 0
        self.assertTrue(
            lab.all(self.regularized_model.grad(self.X, self.y)[:, :, 0] == 0),
            "The min-norm subgradient should be sparse.",
        )

        self.regularized_model.regularizer.lam = 100
        # test the gradient against finite differences when all weights are non-zero
        for i in range(self.tries):
            v = (
                self.rng.standard_normal(
                    (self.c * self.p * self.d), dtype=self.dtype
                )
                * 1
            )

            # we expect a lot of numerical error in finite-difference approximation.
            self.assertTrue(
                np.isclose(
                    check_grad(self.objective_fn, self.grad_fn, v),
                    0,
                    rtol=1e-3,
                    atol=1e-3,
                ),
                "Regularized gradient does not match finite differences approximation.",
            )


if __name__ == "__main__":
    unittest.main()
