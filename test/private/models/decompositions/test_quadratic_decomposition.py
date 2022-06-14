"""
Tests for one-sided quadratic model for approximate cone decompositions.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from scnn import activations
from scnn.private.models.decompositions import QuadraticDecomposition
from scnn.private.utils.data import gen_regression_data
from scnn.private.models import L2Regularizer, GroupL1Regularizer


@parameterized_class(lab.TEST_GRID)
class TestIneqAugmentedConvexMLP(unittest.TestCase):
    """Test convex formulation of two-layer ReLU network with inequality constraints."""

    d: int = 2
    n: int = 4
    c: int = 5
    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(
            self.rng, self.n, 0, self.d, c=self.c
        )
        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.P = self.D.shape[1]

        self.X, _ = lab.all_to_tensor(train_set)
        self.y = lab.tensor(
            self.rng.normal(0, 1, size=(self.c, self.P, self.n))
        )

        self.regularizer = L2Regularizer(0.01)

        self.decomp = QuadraticDecomposition(
            self.d,
            self.D,
            c=self.c,
            regularizer=self.regularizer,
        )

    def test_combined_grad(self):
        """Test implementation of objective and gradient decomposition for
        combined objective."""

        self.decomp.combined = True

        def obj_fn(w):
            return self.decomp.objective(
                self.X,
                self.y,
                lab.tensor(w),
            )

        def grad_fn(w):
            return lab.to_np(
                self.decomp.grad(
                    self.X,
                    self.y,
                    lab.tensor(w),
                    flatten=True,
                )
            )

        for tr in range(self.tries):
            weights = self.rng.standard_normal(
                (self.c, self.P, self.d), dtype=self.dtype
            )

            self.assertTrue(
                np.allclose(
                    check_grad(obj_fn, grad_fn, weights.reshape(-1)),
                    0.0,
                    atol=1e-4,
                ),
                "The gradient of the objective does not match the finite-difference approximation.",
            )

    def test_separable_grad(self):
        """Test implementation of objective and gradient decomposition for
        separable objective."""

        self.decomp.combined = False

        def obj_fn(w):
            return self.decomp.objective(
                self.X,
                self.y,
                lab.tensor(w),
            )

        def grad_fn(w):
            return lab.to_np(
                self.decomp.grad(
                    self.X,
                    self.y,
                    lab.tensor(w),
                    flatten=True,
                )
            )

        for tr in range(self.tries):
            weights = self.rng.standard_normal(
                (self.c, self.P, self.d), dtype=self.dtype
            )

            self.assertTrue(
                np.allclose(
                    check_grad(obj_fn, grad_fn, weights.reshape(-1)),
                    0.0,
                    atol=1e-4,
                ),
                "The gradient of the objective does not match the finite-difference approximation.",
            )


if __name__ == "__main__":
    unittest.main()
