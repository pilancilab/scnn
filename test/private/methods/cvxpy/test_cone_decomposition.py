"""
Tests for convex neural networks.
"""
from typing import cast
import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from scnn import activations
from scnn.private.models import ConvexMLP, AL_MLP
from scnn.private.utils.data import gen_regression_data

from scnn.private.methods.cvxpy import (
    MinL2Decomposition,
    MinL1Decomposition,
    FeasibleDecomposition,
    MinRelaxedL2Decomposition,
    SOCPDecomposition,
)


@parameterized_class(lab.TEST_GRID)
class TestDecompositionPrograms(unittest.TestCase):
    """Test convex formulation of two-layer ReLU network."""

    d: int = 10
    n: int = 5
    c: int = 3
    rng: np.random.Generator = np.random.default_rng(778)

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(self.rng, self.n, 0, self.d, c=self.c)
        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)

        self.P = self.D.shape[1]

        self.model = ConvexMLP(self.d, self.D, self.U, kernel="einsum", c=self.c)

    def test_l2_decomposition(self):
        self.model.weights = lab.tensor(
            self.rng.standard_normal((self.c, self.P, self.d), dtype=self.dtype)
        )

        decomposition_program = MinL2Decomposition("ecos")

        relu_model, exit_status = decomposition_program(self.model, self.X, self.y)
        relu_model = cast(AL_MLP, relu_model)

        # check for feasibility
        e_gap, i_gap = relu_model.constraint_gaps(self.X)

        self.assertTrue(
            exit_status["success"],
            "Cone decomposition did not succeed!",
        )

        self.assertTrue(
            lab.allclose(lab.sum(lab.abs(i_gap)), lab.tensor(0.0)),
            "Result of decomposition does not satisfy cone constraints!",
        )

        self.assertTrue(
            lab.allclose(relu_model.get_reduced_weights(), self.model.weights),
            "Result of decomposition is not equivalent to original model!",
        )

    def test_relaxed_l2_decomposition(self):
        self.model.weights = lab.tensor(
            self.rng.standard_normal((self.c, self.P, self.d), dtype=self.dtype)
        )

        decomposition_program = MinRelaxedL2Decomposition("ecos")

        relu_model, exit_status = decomposition_program(self.model, self.X, self.y)
        relu_model = cast(AL_MLP, relu_model)

        # check for feasibility
        e_gap, i_gap = relu_model.constraint_gaps(self.X)

        self.assertTrue(
            exit_status["success"],
            "Cone decomposition did not succeed!",
        )

        self.assertTrue(
            lab.allclose(
                lab.sum(lab.abs(i_gap)), lab.tensor(0.0), rtol=1e-6, atol=1e-6
            ),
            "Result of decomposition does not satisfy cone constraints!",
        )

        self.assertTrue(
            lab.allclose(relu_model.get_reduced_weights(), self.model.weights),
            "Result of decomposition is not equivalent to original model!",
        )

    def test_l1_decomposition(self):
        self.model.weights = lab.tensor(
            self.rng.standard_normal((self.c, self.P, self.d), dtype=self.dtype)
        )

        decomposition_program = MinL1Decomposition("ecos")

        relu_model, exit_status = decomposition_program(self.model, self.X, self.y)
        relu_model = cast(AL_MLP, relu_model)

        # check for feasibility
        e_gap, i_gap = relu_model.constraint_gaps(self.X)

        self.assertTrue(
            exit_status["success"],
            "Cone decomposition did not succeed!",
        )

        self.assertTrue(
            lab.allclose(
                lab.sum(lab.abs(i_gap)), lab.tensor(0.0), rtol=1e-6, atol=1e-6
            ),
            "Result of decomposition does not satisfy cone constraints!",
        )

        self.assertTrue(
            lab.allclose(relu_model.get_reduced_weights(), self.model.weights),
            "Result of decomposition is not equivalent to original model!",
        )

    def test_feasible_decomposition(self):
        self.model.weights = lab.tensor(
            self.rng.standard_normal((self.c, self.P, self.d), dtype=self.dtype)
        )

        decomposition_program = FeasibleDecomposition("ecos")

        relu_model, exit_status = decomposition_program(self.model, self.X, self.y)
        relu_model = cast(AL_MLP, relu_model)

        # check for feasibility
        e_gap, i_gap = relu_model.constraint_gaps(self.X)

        self.assertTrue(
            exit_status["success"],
            "Cone decomposition did not succeed!",
        )

        self.assertTrue(
            lab.allclose(lab.sum(lab.abs(i_gap)), lab.tensor(0.0)),
            "Result of decomposition does not satisfy cone constraints!",
        )

        self.assertTrue(
            lab.allclose(relu_model.get_reduced_weights(), self.model.weights),
            "Result of decomposition is not equivalent to original model!",
        )

    def test_socp_decomposition(self):
        self.model.weights = lab.tensor(
            self.rng.standard_normal((self.c, self.P, self.d), dtype=self.dtype)
        )

        decomposition_program = SOCPDecomposition("ecos")

        relu_model, exit_status = decomposition_program(self.model, self.X, self.y)
        relu_model = cast(AL_MLP, relu_model)

        # check for feasibility
        e_gap, i_gap = relu_model.constraint_gaps(self.X)

        self.assertTrue(
            exit_status["success"],
            "Cone decomposition did not succeed!",
        )

        self.assertTrue(
            lab.allclose(lab.sum(lab.abs(i_gap)), lab.tensor(0.0)),
            "Result of decomposition does not satisfy cone constraints!",
        )

        self.assertTrue(
            lab.allclose(relu_model.get_reduced_weights(), self.model.weights),
            "Result of decomposition is not equivalent to original model!",
        )


if __name__ == "__main__":
    unittest.main()
