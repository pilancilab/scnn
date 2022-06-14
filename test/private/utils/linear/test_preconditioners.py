"""
Test preconditioners.
"""
import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from scnn import activations
from scnn.private.models.convex import operators
from scnn.private.utils.linear import preconditioners
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestPreconditioners(unittest.TestCase):
    """Test preconditioners for iterative solvers."""

    d: int = 2
    n: int = 4
    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(self.rng, self.n, 0, self.d)
        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)

        self.P = self.D.shape[1]

    def test_column_norm(self):
        """Test preconditioner that normalizes columns of X."""

        # Test on original features, X.
        forward = preconditioners.column_norm(self.X)

        # extract preconditioned X by applying forward to identity matrix.
        preconditioned_X = lab.matmul(self.X, forward.matmat(lab.eye(self.d)))
        column_norms = lab.sum(preconditioned_X ** 2, axis=0)
        self.assertTrue(
            lab.allclose(column_norms, lab.ones_like(column_norms)),
            "Preconditioned matrix did not have unit column norms.",
        )

    def test_column_norm_expanded(self):
        """Test preconditioner that normalizes columns of X."""

        # Test on original features, X.
        forward = preconditioners.column_norm(self.X, self.D)
        expanded_X = operators.expanded_data_matrix(self.X, self.D)
        # extract preconditioned X by applying forward to identity matrix.
        preconditioned_X = lab.matmul(
            expanded_X, forward.matmat(lab.eye(self.d * self.P))
        )
        column_norms = lab.sum(preconditioned_X ** 2, axis=0)
        self.assertTrue(
            lab.allclose(column_norms, lab.ones_like(column_norms)),
            "Preconditioned matrix did not have unit column norms.",
        )


if __name__ == "__main__":
    unittest.main()
