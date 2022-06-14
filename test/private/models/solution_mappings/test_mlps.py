"""
Tests for convex neural networks.
"""

import unittest

import torch
import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from scnn import activations
from scnn.private.models import (
    ConvexMLP,
    AL_MLP,
    GatedReLULayer,
)
import scnn.private.models.solution_mappings.mlps as sm
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestSolutionMappings(unittest.TestCase):
    """Test mappings between solutions to the convex and non-convex problems."""

    d: int = 2
    n: int = 4
    c: int = 5
    rng: np.random.Generator = np.random.default_rng(779)

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(
            self.rng, self.n, 0, self.d, c=self.c
        )

        train_set = (
            np.concatenate([train_set[0], lab.ones((self.n, 1))], axis=1),
            train_set[1],
        )

        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(
                train_set[0], self.U, bias=True, active_proportion=0.5
            )
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.d = self.d + 1

        self.P = self.D.shape[1]

        self.networks = {}
        self.gated_convex_mlp = ConvexMLP(
            self.d, self.D, self.U, kernel="einsum", c=self.c
        )
        self.relu_convex_mlp = AL_MLP(
            self.d, self.D, self.U, kernel="einsum", c=self.c
        )

    def test_is_relu_compatible(self):
        """Test compatibility for ReLU models."""

        # try several invalid architectures.
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.Sigmoid(), torch.nn.Linear(10, 1)
        )
        self.assertFalse(
            sm.is_relu_compatible(torch_model),
            "Sigmoids activations should not yield valid architectures for convex formulations.",
        )
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )
        self.assertFalse(
            sm.is_relu_compatible(torch_model),
            "Bias variables are not permitted and should not yield valid architectures for convex formulations.",
        )
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
        )
        self.assertFalse(
            sm.is_relu_compatible(torch_model),
            "Networks which have too many layers should not yield valid architectures for convex formulations.",
        )

        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1, bias=False),
        )
        self.assertTrue(
            sm.is_relu_compatible(torch_model),
            "A two-layer network with ReLU activations and without biases should be a valid architecture.",
        )

    def test_is_grelu_compatible(self):
        """Test compatibility for Gated ReLU models."""

        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1, bias=False),
        )
        self.assertFalse(
            sm.is_grelu_compatible(torch_model),
            "A two-layer network with ReLU activations and without biases should not be a valid architecture for the relaxed problem.",
        )

        torch_model = torch.nn.Sequential(
            GatedReLULayer(self.U),
            torch.nn.Linear(self.P, 1, bias=False),
        )
        self.assertTrue(
            sm.is_grelu_compatible(torch_model),
            "A two-layer network with gated ReLU activations and without biases should be a valid architecture for the relaxed problem.",
        )

    def test_mapping_gated_models_to_nonconvex(self):

        self.gated_convex_mlp.weights = lab.tensor(
            self.rng.random(self.gated_convex_mlp.weights.shape)
        )

        torch_model = sm.construct_nc_torch(
            self.gated_convex_mlp, grelu=True, remove_sparse=False
        )
        manual_model = sm.construct_nc_manual(
            self.gated_convex_mlp, grelu=True, remove_sparse=False
        )

        X = lab.torch_backend.torch_tensor(self.X)

        # check that predictions are identical for the two models
        self.assertTrue(
            torch.allclose(
                torch_model(X),
                lab.torch_backend.torch_tensor(self.gated_convex_mlp(X)),
            ),
            "The PyTorch version of the model did not have the same predictions!",
        )
        self.assertTrue(
            lab.allclose(manual_model(X), self.gated_convex_mlp(X)),
            "The manual version of the model did not have the same predictions!",
        )

        # test sparse models
        self.gated_convex_mlp.weights[:, [0, 2]] = 0.0

        torch_model = sm.construct_nc_torch(
            self.gated_convex_mlp, grelu=True, remove_sparse=True
        )
        manual_model = sm.construct_nc_manual(
            self.gated_convex_mlp, grelu=True, remove_sparse=True
        )

        X = lab.torch_backend.torch_tensor(self.X)

        # check that predictions are identical for the two models
        self.assertTrue(
            torch.allclose(
                torch_model(X),
                lab.torch_backend.torch_tensor(self.gated_convex_mlp(X)),
            ),
            "The PyTorch version of the model did not have the same predictions!",
        )
        self.assertTrue(
            lab.allclose(manual_model(X), self.gated_convex_mlp(X)),
            "The manual version of the model did not have the same predictions!",
        )

    def test_mapping_relu_models_to_nonconvex(self):

        # set weights to be an interior point of the constraint set.
        weights = lab.stack([c * self.U.T for c in range(self.c)], axis=0)
        self.relu_convex_mlp.weights = lab.stack([weights, 0.1 * weights])

        torch_model = sm.construct_nc_torch(
            self.relu_convex_mlp, grelu=False, remove_sparse=False
        )
        manual_model = sm.construct_nc_manual(
            self.relu_convex_mlp, grelu=False, remove_sparse=False
        )

        X = lab.torch_backend.torch_tensor(self.X)

        # check that predictions are identical for the two models
        self.assertTrue(
            torch.allclose(
                torch_model(X),
                lab.torch_backend.torch_tensor(self.relu_convex_mlp(X)),
            ),
            "The PyTorch version of the model did not have the same predictions!",
        )
        self.assertTrue(
            lab.allclose(manual_model(X), self.relu_convex_mlp(X)),
            "The manual version of the model did not have the same predictions!",
        )

        # test sparse models
        self.relu_convex_mlp.weights[0, :, [0, 2]] = 0.0
        self.relu_convex_mlp.weights[0, :, [1]] = 0.0

        torch_model = sm.construct_nc_torch(
            self.relu_convex_mlp, grelu=False, remove_sparse=True
        )
        manual_model = sm.construct_nc_manual(
            self.relu_convex_mlp, grelu=False, remove_sparse=True
        )

        X = lab.torch_backend.torch_tensor(self.X)

        # check that predictions are identical for the two models
        self.assertTrue(
            torch.allclose(
                torch_model(X),
                lab.torch_backend.torch_tensor(self.relu_convex_mlp(X)),
            ),
            "The PyTorch version of the model did not have the same predictions!",
        )
        self.assertTrue(
            lab.allclose(manual_model(X), self.relu_convex_mlp(X)),
            "The manual version of the model did not have the same predictions!",
        )


if __name__ == "__main__":
    unittest.main()
