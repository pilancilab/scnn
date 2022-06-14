"""
Tests for proximal operators.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore
import lab

from scnn.private.prox import (
    L1,
    GroupL1,
    FeatureGroupL1,
    Orthant,
    GroupL1Orthant,
)


@parameterized_class(lab.TEST_GRID)
class TestProximalOps(unittest.TestCase):
    """Test proximal operators."""

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

    def test_l1(self):
        """Test proximal operator for l1 penalty."""

        # if lambda = 0, then the prox is the identity operator.
        vector = 1.0 + lab.arange(10)
        matrix = 1.0 + lab.arange(100).reshape(10, 10)
        prox = L1(lam=0)

        self.assertTrue(
            lab.allclose(prox(vector, 1), vector),
            "Prox with lambda = 0 didn't reduce to the identity map for vectors.",
        )
        self.assertTrue(
            lab.allclose(prox(matrix, 1), matrix),
            "Prox with lambda = 0 didn't reduce to the identity map for matrices.",
        )

        # if lambda is the max entry, then all entries are zeroed
        prox.lam = 10
        self.assertTrue(
            lab.all(prox(vector, 1) == 0),
            "Prox with lambda = max(w) didn't zero all entries.",
        )
        prox.lam = 100
        self.assertTrue(
            lab.all(prox(matrix, 1) == 0),
            "Prox with lambda = max(w) didn't zero all entries.",
        )

        # if lambda is < max entry, then at least one entry is non-zero
        prox.lam = 9
        self.assertTrue(
            lab.any(prox(vector, 1) != 0),
            "Prox with lambda = max(w) didn't zero all entries.",
        )
        prox.lam = 99
        self.assertTrue(
            lab.any(prox(matrix, 1) != 0),
            "Prox with lambda = max(w) didn't zero all entries.",
        )

    def test_group_l1(self):
        """Test proximal operator for group l1 penalty."""
        matrix = lab.tensor(
            [
                [
                    [
                        [-1.0, -10, -100, -1000],
                        [-1, -10, -100, -1000],
                        [-1, -10, -100, -1000],
                    ]
                ],
                [
                    [
                        [1.0, 10, 100, 1000],
                        [1, 10, 100, 1000],
                        [1, 10, 100, 1000],
                    ]
                ],
            ]
        )
        prox = FeatureGroupL1(lam=0)

        # if lambda is 0, then the prox is the identity operator
        self.assertTrue(
            lab.allclose(prox(matrix, 1), matrix),
            "Prox with lambda = 0 didn't reduce to the identity map for matrices.",
        )

        # if lambda is huge, then all entries are zeroed
        prox.lam = np.sqrt(6) * 1000
        self.assertTrue(
            lab.all(prox(matrix, 1) == 0),
            "Prox with lambda very large didn't zero all matrix entries.",
        )

        # if lambda is large, then some entries should be zerod
        prox.lam = np.sqrt(6) * 100
        zeroed_matrix = prox(matrix, 1)
        self.assertTrue(
            lab.all(zeroed_matrix[:, :, :, 0:3] == 0.0),
            "The first three rows didn't get zeroed.",
        )
        self.assertTrue(
            lab.all(zeroed_matrix[:, :, :, 3] != 0.0),
            "The last row had zero elements.",
        )

        # if lambda is too small, then no entries are zeroed
        prox.lam = 1
        zeroed_matrix = prox(matrix, 1)
        self.assertTrue(
            lab.all(zeroed_matrix != 0), "All entires should be non-zero."
        )

    def test_group_l1(self):
        """Test proximal operator for group l1 penalty."""
        matrix = lab.tensor(
            [
                [
                    [
                        [-1.0, -10, -100, -1000],
                        [-1, -10, -100, -1000],
                        [-1, -10, -100, -1000],
                    ]
                ],
                [
                    [
                        [1.0, 10, 100, 1000],
                        [1, 10, 100, 1000],
                        [1, 10, 100, 1000],
                    ]
                ],
            ]
        )
        matrix = lab.transpose(matrix, 3, 2)

        prox = GroupL1(lam=0)

        # if lambda is 0, then the prox is the identity operator
        self.assertTrue(
            lab.allclose(prox(matrix, 1), matrix),
            "Prox with lambda = 0 didn't reduce to the identity map for matrices.",
        )

        # if lambda is large, then all entries are zeroed
        prox.lam = np.sqrt(3) * 1000
        self.assertTrue(
            lab.all(prox(matrix, 1) == 0),
            "Prox with lambda very large didn't zero all matrix entries.",
        )

        # if lambda is large, then all entries are zeroed
        prox.lam = np.sqrt(3) * 100
        zeroed_matrix = prox(matrix, 1)
        self.assertTrue(
            lab.all(zeroed_matrix[:, :, 0:2, :] == 0.0),
            "The first three rows didn't get zeroed.",
        )
        self.assertTrue(
            lab.all(zeroed_matrix[:, :, 3, :] != 0.0),
            "The last row had zero elements.",
        )

        # if lambda is too small, then no entries are zeroed
        prox.lam = 1
        zeroed_matrix = prox(matrix, 1)
        self.assertTrue(
            lab.all(zeroed_matrix != 0), "All entires should be non-zero."
        )

    def test_orthant(self):
        """Test projecting onto orthants."""

        # generate random orthants
        k, n, d, c = 2, 5, 10, 7
        rng = np.random.default_rng(255)
        A = lab.tensor(rng.choice([1, -1], size=(k, n, d), replace=True))
        orthant_proj = Orthant(A)

        # project random vectors onto the orthants
        z = lab.tensor(rng.standard_normal((k, c, d, n), dtype=self.dtype))
        x = orthant_proj(z)

        # check KKT conditions
        nu = -lab.smin(lab.einsum("ikj, imjk->imjk", A, z), 0)
        Ax = lab.einsum("ikj, imjk->imjk", A, x)
        Anu = lab.einsum("ikj, imjk->imjk", A, nu)

        self.assertTrue(lab.all(Ax >= 0), "The projection is not feasible!")
        self.assertTrue(
            lab.allclose(x, z + Anu), "The Lagrangian is not stationary is x!"
        )
        slacks = lab.sum(lab.multiply(Ax, nu), axis=1)
        self.assertTrue(
            lab.allclose(slacks, lab.zeros_like(slacks)),
            "Complementary slackness failed!",
        )

    def test_groupl1_orthant(self):
        """Test joint group-l1/orthant constraints."""

        # generate random orthants
        k, d, n, p, c = 2, 5, 20, 10, 7
        rng = np.random.default_rng(255)

        A = lab.tensor(rng.choice([1, -1], size=(k, n, p), replace=True))
        lam = 1
        orthant_proj = Orthant(A)
        group_prox = GroupL1(lam)
        combined_op = GroupL1Orthant(d, lam, A)

        # project random vectors onto the orthants
        z = lab.tensor(rng.standard_normal((k, c, p, d + n), dtype=self.dtype))

        x1 = group_prox(z[:, :, :, :d], beta=1)
        x2 = orthant_proj(z[:, :, :, d:], beta=1)

        x = combined_op(z, beta=1)

        self.assertTrue(
            lab.allclose(x, lab.concatenate([x1, x2], axis=-1)),
            "Individual computation of the group L1/orthant projection operator did not match joint computation.",
        )


if __name__ == "__main__":
    unittest.main()
