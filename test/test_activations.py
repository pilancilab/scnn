"""Test different methods for sampling activation patterns.

TODO:
    - 
"""

import unittest

import numpy as np

from scnn.private.utils.data.synthetic import gen_regression_data
from scnn.activations import (
    sample_gate_vectors,
    sample_sparse_gates,
    generate_index_lists,
    compute_activation_patterns,
)


class TestActivations(unittest.TestCase):
    """Test different methods for sampling activation patterns."""

    def setUp(self):
        # Generate realizable synthetic classification problem (ie. Figure 1)
        n_train = 100
        n_test = 100
        self.d = 25
        c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(123, n_train, n_test, self.d, c, kappa=kappa)
        self.seed = 123
        self.n_gates = 1000

    def test_dense_gates(self):
        """Test sampling dense gate vectors."""

        G = sample_gate_vectors(
            self.seed, self.d, self.n_gates, gate_type="dense"
        )

        # check gate shape.
        self.assertTrue(G.shape == (self.d, self.n_gates))

        # check gates are dense
        self.assertTrue(np.all(G != 0.0))

    def test_feature_sparse_gates(self):
        """Test sampling feature-sparse gate vectors."""
        G = sample_sparse_gates(
            np.random.default_rng(self.seed),
            self.d,
            self.n_gates,
            sparsity_indices=np.arange(self.d).reshape(self.d, 1).tolist(),
        )

        # check gate shape.
        self.assertTrue(G.shape == (self.d, self.n_gates))

        # each gate should have exactly one non-zero index
        self.assertTrue(np.all(np.sum(G != 0, axis=0) == 1))

        # try generating index lists
        order_one = generate_index_lists(self.d, 1)
        self.assertTrue(
            np.all(order_one == np.arange(self.d).reshape(self.d, 1).tolist())
        )

        order_two = generate_index_lists(self.d, 2)

        G = sample_sparse_gates(
            np.random.default_rng(self.seed),
            self.d,
            self.n_gates,
            sparsity_indices=order_two,
        )

        # each gate should have exactly one or two non-zero indices
        n_sparse_indices = np.sum(G != 0, axis=0)
        self.assertTrue(
            np.all(np.logical_or(n_sparse_indices == 1, n_sparse_indices == 2))
        )

    def test_bias_terms(self):
        """Test sampling gate vectors with bias terms."""

        # manually compute activations:
        def activations(X, G):
            XG = np.matmul(X, G)
            XG = np.maximum(XG, 0)
            XG[XG > 0] = 1
            return XG

        # augment data with bias term
        X = np.concatenate(
            [self.X_train, np.ones((self.X_train.shape[0], 1))], axis=1
        )

        G = sample_gate_vectors(
            123,
            self.d,
            self.n_gates,
            gate_type="dense",
        )

        D, G_bias = compute_activation_patterns(
            X, G, bias=True, active_proportion=0.5
        )

        self.assertTrue(
            np.all(np.sum(D, axis=0) == X.shape[0] / 2),
            "Half of the examples should be active for each hyperplane",
        )

        D_manual = activations(X, G_bias)
        self.assertTrue(
            np.all(D_manual == D), "The activations should be produced by G."
        )

        D, G_bias = compute_activation_patterns(
            X, G, bias=True, active_proportion=0.1
        )

        self.assertTrue(
            np.all(np.sum(D, axis=0) == X.shape[0] / 10),
            "1/10 of the examples should be active for each hyperplane",
        )

        D_manual = activations(X, G_bias)
        self.assertTrue(
            np.all(D_manual == D), "The activations should be produced by G."
        )


if __name__ == "__main__":
    unittest.main()
