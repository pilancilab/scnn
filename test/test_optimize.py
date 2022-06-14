"""End-to-end tests for :module:`scnn.optimize`.

These tests exercise the internal models and optimization routines in :module:`scnn.private` as well as the
interface between the public and private modules :module`scnn.private.interface`.

Most tests are end-to-end and don't check specific termination conditions. Rather, they ensure that the entire 
library works properly when invoked with particular inputs.


TODO:
    -
"""

import unittest
import torch
import numpy as np

from scnn.private.utils.data.synthetic import gen_regression_data

from scnn.optimize import optimize, optimize_model, optimize_path
from scnn.regularizers import NeuronGL1
from scnn.models import ConvexGatedReLU, ConvexReLU
from scnn.solvers import RFISTA, AL
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors


class TestDataFormats(unittest.TestCase):
    """Test inputting different data formats different floating-point precisions.

    We support three data formats at the moment:

    1. Python lists and lists of lists.
    2. NumPy arrays
    3. PyTorch tensors located on the CPU device.
    """

    def setUp(self):
        # Generate realizable synthetic classification problem (ie. Figure 1)
        n_train = 100
        n_test = 100
        d = 25
        c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(123, n_train, n_test, d, c, kappa=kappa)

        self.regularizer = NeuronGL1(0.0001)

    def test_scalar_targets(self):
        """Test :func:`scnn.optimize.optimize` with scalar targets."""

        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train[:, 0],
            self.X_test,
            self.y_test[:, 0],
            self.regularizer,
        )

    def test_numpy(self):
        """Test :func:`scnn.optimize.optimize` with data as NumPy ndarrays."""

        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

    def test_lists(self):
        """Test :func:`scnn.optimize.optimize` with data in list of lists format."""

        X_train = self.X_train.tolist()
        X_test = self.X_test.tolist()

        y_train = self.y_train.tolist()
        y_test = self.y_test.tolist()

        # train model
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            X_train,
            y_train,
            X_test,
            y_test,
            self.regularizer,
        )

    def test_torch(self):
        """Test :func:`scnn.optimize.optimize` with data as PyTorch tensors."""
        X_train = torch.tensor(self.X_train.tolist())
        X_test = torch.tensor(self.X_test.tolist())

        y_train = torch.tensor(self.y_train.tolist())
        y_test = torch.tensor(self.y_test.tolist())

        # train model
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            X_train,
            y_train,
            X_test,
            y_test,
            self.regularizer,
        )

    def test_no_test_data(self):
        """Test :func:`scnn.optimize.optimize` without any test data."""
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            regularizer=self.regularizer,
        )


class TestFunctionalInterface(unittest.TestCase):
    """Test simple functional interface to `scnn`."""

    def setUp(self):
        # Generate realizable synthetic classification problem (ie. Figure 1)
        n_train = 100
        n_test = 100
        d = 25
        c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(123, n_train, n_test, d, c, kappa=kappa)

        self.regularizer = NeuronGL1(0.0001)

    def test_solving_gated_relu(self):
        """Test solving the Gated ReLU problem with neuron-wise GL1 regularization using
        the default solver, R-FISTA.
        """
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

    def test_solving_relu(self):
        """Test solving the ReLU problem with neuron-wise GL1 regularization using
        the default solver, AL method with R-FISTA as a sub-routine.
        """
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )


class TestOOInterface(unittest.TestCase):
    """Test object-oriented interface to `scnn`."""

    def setUp(self):

        n_train = 100
        n_test = 100
        self.d = 5
        self.c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(
            123, n_train, n_test, self.d, self.c, kappa=kappa
        )
        self.metrics = Metrics(
            metric_freq=25,
            model_loss=True,
            train_accuracy=True,
            train_mse=True,
            test_mse=True,
            test_accuracy=True,
            neuron_sparsity=True,
        )
        self.lam = 0.0001
        self.regularizer = NeuronGL1(self.lam)
        self.G = sample_gate_vectors(
            np.random.default_rng(123), self.d, self.max_neurons
        )

        # regularization paths
        self.lam_path = np.logspace(-5, -2, 3)
        self.path = [NeuronGL1(lam) for lam in self.lam_path]

    def test_solving_gated_relu(self):
        """Test solving the Gated ReLU problem with neuron-wise GL1 regularization using
        the default solver, R-FISTA.
        """

        # Instantiate convex model and other options.
        model = ConvexGatedReLU(self.G, c=self.c)
        solver = RFISTA(model, tol=1e-6)

        oo_model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

        # check against functional interface
        f_model, _ = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
            seed=123,  # ensures the same activation patterns are used.
        )

        self.assertTrue(
            np.allclose(
                oo_model.get_parameters()[0], f_model.get_parameters()[0]
            )
        )

    def test_solving_gated_relu_path(self):
        """Test solving a regularization path for the Gated ReLU problem."""

        # Instantiate convex model and other options.
        model = ConvexGatedReLU(self.G, c=self.c)
        solver = RFISTA(model, tol=1e-6)

        model_path, _ = optimize_path(
            model,
            solver,
            self.path,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            warm_start=True,
            save_path="test/test_data/gated",
        )

    def test_solving_relu(self):
        """Test solving the ReLU problem with neuron-wise GL1 regularization using
        the default solver, AL method with R-FISTA as a sub-routine.
        """

        model = ConvexReLU(self.G, c=self.c)
        solver = AL(model)

        oo_model, _ = optimize_model(
            model,
            solver,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
        )

        # check against functional interface
        f_model, _ = optimize(
            "relu",
            2 * self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.regularizer,
            seed=123,  # ensures the same activation patterns are used.
        )

        self.assertTrue(
            np.allclose(
                oo_model.get_parameters()[0], f_model.get_parameters()[0]
            )
        )

        self.assertTrue(
            np.allclose(
                oo_model.get_parameters()[1], f_model.get_parameters()[1]
            )
        )

    def test_solving_relu_path(self):
        """Test solving a regularization path for the ReLU problem."""

        # Instantiate convex model and other options.
        model = ConvexReLU(self.G, c=self.c)
        solver = AL(model)

        model_path, _ = optimize_path(
            model,
            solver,
            self.path,
            self.metrics,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            warm_start=True,
            save_path="test/test_data/relu",
        )


if __name__ == "__main__":
    unittest.main()
