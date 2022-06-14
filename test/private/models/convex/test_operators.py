"""
Tests for "kernel" functions and matrix builders for convex neural networks.
"""

import unittest
from itertools import product

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from scnn import activations
import scnn.private.loss_functions as loss_fns
from scnn.private.utils.data import gen_regression_data
from scnn.private.models.convex import operators


@parameterized_class(lab.TEST_GRID)
class TestExpandedModelKernels(unittest.TestCase):
    """Test operators and matrix builders for the expanded linear model corresponding to a convex neural network."""

    d: int = 2
    n: int = 3
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
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)

        self.P = self.D.shape[1]
        self.kernels = {}
        self.builders = {}

        for kernel_name in operators.KERNELS:
            (
                data_mvp,
                data_t_mvp,
                gradient,
                hessian_mvp,
                bd_hessian_mvp,
            ) = operators.get_kernel(kernel_name)

            self.kernels[kernel_name] = {
                "data_mvp": data_mvp,
                "data_t_mvp": data_t_mvp,
                "gradient": gradient,
                "hessian_mvp": hessian_mvp,
                "bd_hessian_mvp": bd_hessian_mvp,
            }
            data, hessian, bd_hessian = operators.get_matrix_builders(
                kernel_name
            )

            self.builders[kernel_name] = {
                "data": data,
                "hessian": hessian,
                "bd_hessian": bd_hessian,
            }

    def test_kernels_finite_diff(self):
        """Test implementations of gradient/Hessian kernels for the expanded linear model against finite-difference calculations."""
        # check the gradient against finite differences
        for i, kernel_name in product(range(self.tries), self.kernels.keys()):

            v = self.rng.standard_normal(
                (self.c * self.P * self.d), dtype=self.dtype
            )

            def loss_fn(v):
                res = lab.to_np(
                    loss_fns.squared_error(
                        self.kernels[kernel_name]["data_mvp"](
                            lab.tensor(v, dtype=lab.get_dtype()),
                            self.X,
                            self.D,
                        ),
                        self.y,
                    )
                )
                return res

            def grad_fn(v):
                res = lab.to_np(
                    2
                    * self.kernels[kernel_name]["gradient"](
                        lab.tensor(v, dtype=lab.get_dtype()),
                        X=self.X,
                        y=self.y,
                        D=self.D,
                    )
                )
                return res

            self.assertTrue(
                np.isclose(
                    check_grad(loss_fn, grad_fn, v), 0.0, rtol=1e-5, atol=1e-5
                ),
                f"Gradient from kernel {kernel_name} did not match finite differences approximation.",
            )

            # check Hessian against finite differences by sampling random directional derivatives
            v = self.rng.standard_normal(
                self.c * self.P * self.d, dtype=self.dtype
            )
            z = lab.tensor(
                self.rng.standard_normal(
                    self.c * self.P * self.d, dtype=self.dtype
                )
            )

            def directional_derivative(w):
                return lab.to_np(
                    self.kernels[kernel_name]["gradient"](
                        lab.tensor(w), X=self.X, y=self.y, D=self.D
                    )
                    @ z
                )

            def hvp(w):
                return lab.to_np(
                    self.kernels[kernel_name]["hessian_mvp"](
                        z, X=self.X, D=self.D
                    )
                )

            self.assertTrue(
                np.isclose(
                    check_grad(directional_derivative, hvp, v),
                    0,
                    rtol=1e-5,
                    atol=1e-5,
                ),
                f"Hessian from kernel {kernel_name} did not match finite differences approximation.",
            )

    def test_kernels_against_reference(self):
        """Test kernel implementations against reference implementation (direct calculation)."""

        # check all kernels against reference implementation (direct).
        for i, kernel_name in product(range(self.tries), self.kernels.keys()):
            v = lab.tensor(
                self.rng.standard_normal(
                    (self.c, self.P, self.d), dtype=self.dtype
                )
            )

            # check the forward map (ie. X w):
            self.assertTrue(
                lab.allclose(
                    self.kernels[kernel_name]["data_mvp"](v, self.X, self.D),
                    self.kernels[operators.DIRECT]["data_mvp"](
                        v, self.X, self.D
                    ),
                ),
                f"The {kernel_name} matrix-vector operator for the expanded data matrix did not match the direct implementation.",
            )

            # check transpose
            w = lab.tensor(self.rng.standard_normal(self.n, dtype=self.dtype))
            self.assertTrue(
                lab.allclose(
                    self.kernels[kernel_name]["data_t_mvp"](w, self.X, self.D),
                    self.kernels[operators.DIRECT]["data_t_mvp"](
                        w, self.X, self.D
                    ),
                ),
                f"The {kernel_name} matrix-vector operator for the transpose of the expanded data matrix did not match the direct implementation.",
            )

            # check gradient
            self.assertTrue(
                lab.allclose(
                    self.kernels[kernel_name]["gradient"](
                        v, self.X, self.y, self.D
                    ),
                    self.kernels[operators.DIRECT]["gradient"](
                        v, self.X, self.y, self.D
                    ),
                ),
                f"The {kernel_name} gradient operator did not match the direct implementation.",
            )

            # check Hessian
            self.assertTrue(
                lab.allclose(
                    self.kernels[kernel_name]["hessian_mvp"](
                        v, self.X, self.D
                    ),
                    self.kernels[operators.DIRECT]["hessian_mvp"](
                        v, self.X, self.D
                    ),
                ),
                f"The {kernel_name} hessian operator did not match the direct implementation.",
            )

            # check block-diagonal Hessian.
            self.assertTrue(
                lab.allclose(
                    self.kernels[kernel_name]["bd_hessian_mvp"](
                        v, self.X, self.D
                    ),
                    self.kernels[operators.DIRECT]["bd_hessian_mvp"](
                        v, self.X, self.D
                    ),
                ),
                f"The {kernel_name} block-diagonal hessian operator did not match the direct implementation.",
            )

    def test_matrix_builders(self):
        """Test matrix builders against reference implementation and finite differences."""
        for kernel_name in self.kernels.keys():

            H = self.builders[kernel_name]["hessian"](self.X, self.D)
            H_flat = lab.concatenate([entry for entry in H], axis=1)
            H_flat = lab.concatenate([entry for entry in H_flat], axis=1)

            bd_H = self.builders[kernel_name]["bd_hessian"](self.X, self.D)

            # compute matrix blocks by hand and check for correctness.
            for i, j in product(range(self.P), range(self.P)):
                H_ij = (
                    self.X.T
                    @ lab.diag(self.D[:, i])
                    @ lab.diag(self.D[:, j])
                    @ self.X
                )

                self.assertTrue(
                    lab.allclose(H_ij, H[i, j]),
                    f"Hessian block from builder {kernel_name} did not match manual computation.",
                )

                if i == j:
                    self.assertTrue(
                        lab.allclose(H_ij, bd_H[i]),
                        f"Diagonal block from builder {kernel_name} did not match manual computation.",
                    )

            # check Hessian against finite differences
            for i in range(self.tries):
                v = self.rng.standard_normal(
                    (self.c * self.P * self.d), dtype=self.dtype
                )

                # check rows of the Hessian independently
                for c in range(self.c):
                    i = 0
                    for j, k in product(range(self.P), range(self.d)):

                        def grad_cjk(w):
                            curr = self.kernels[kernel_name]["gradient"](
                                lab.tensor(w, dtype=lab.get_dtype()),
                                self.X,
                                self.y,
                                self.D,
                            ).reshape((self.c, self.P, self.d))

                            return curr[c, j, k]

                        def hessian_cjk(w):
                            hessian = lab.zeros((self.c, self.P * self.d))
                            hessian[c] = H_flat[i]

                            return hessian.reshape(-1)

                        self.assertTrue(
                            np.isclose(
                                check_grad(grad_cjk, hessian_cjk, v),
                                0,
                                rtol=1e-5,
                                atol=1e-5,
                            ),
                            f"Hessian from builder {kernel_name} did not match finite differences approximation.",
                        )
                        i += 1


if __name__ == "__main__":
    unittest.main()
