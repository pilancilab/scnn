"""
Tests for iterative methods for solving linear systems and least-squares problems.
"""
import unittest
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator  # type: ignore

import lab

from scnn import activations
from scnn.private.models import ConvexMLP
from scnn.private.models.convex import operators
from scnn.private.utils.linear import (
    iterative_solvers,
    direct_solvers,
    preconditioners,
)
from scnn.private.utils.data import gen_regression_data


class TestLSTSQSolvers(unittest.TestCase):
    """Test iterative solvers for least-squares problems."""

    d: int = 5
    n: int = 25

    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(lab.NUMPY)
        train_set, _, _ = gen_regression_data(self.rng, self.n, 0, self.d)
        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)

        self.network = ConvexMLP(self.d, self.D, self.U)
        self.P = self.D.shape[1]

    def test_lstsq_iterative_solve(self):
        """Test solving least squares problems."""

        for lam in [0.0, 1.0, 5]:

            # solve least-squares problem for optimal weights.
            w_ne = direct_solvers.solve_ne(self.X, self.y, lam=lam)

            w_lsmr, exit_status_lsmr = iterative_solvers.lstsq_iterative_solve(
                self.X, self.y, lam=lam, solver=iterative_solvers.LSMR
            )
            w_lsqr, exit_status_lsqr = iterative_solvers.lstsq_iterative_solve(
                self.X, self.y, lam=lam, solver=iterative_solvers.LSQR
            )

            # check that the underlying method converged.
            self.assertTrue(
                exit_status_lsmr["success"],
                f"LSMR solver reported failure to solve system with lam: {lam}.",
            )

            # check that the underlying method converged.
            self.assertTrue(
                exit_status_lsqr["success"],
                f"LSQR solver reported failure to solve system with lam: {lam}.",
            )
            # check that the solutions are the same.
            self.assertTrue(
                np.allclose(w_ne, w_lsmr, rtol=1e-5, atol=1e-5),
                f"The solution to the normal equations differs from the LSMR solution with lam: {lam}.",
            )
            # check that the solutions are the same.
            self.assertTrue(
                np.allclose(w_ne, w_lsqr, rtol=1e-5, atol=1e-5),
                f"The solution to the normal equations differs from the LSQR solution with lam: {lam}.",
            )

    def test_precon_lstsq_iterative_solve(self):
        """Test solving least squares problems with preconditioning."""
        forward = preconditioners.column_norm(self.X)
        for lam in [0.0, 1.0, 5]:

            # solve least-squares problem for optimal weights.

            if lam == 0:
                # solve normally with no regularization to check that precondition doesn't affect solution.
                w_ne = direct_solvers.solve_ne(self.X, self.y, lam=lam)
            else:
                X_tilde = np.divide(self.X, np.sqrt(np.sum(self.X ** 2, axis=0)))
                w_ne = direct_solvers.solve_ne(X_tilde, self.y, lam=lam)
                w_ne = forward.matvec(w_ne)

            w_lsmr, exit_status_lsmr = iterative_solvers.lstsq_iterative_solve(
                aslinearoperator(self.X),
                self.y,
                lam=lam,
                solver=iterative_solvers.LSMR,
                preconditioner=forward,
            )

            w_lsqr, exit_status_lsqr = iterative_solvers.lstsq_iterative_solve(
                aslinearoperator(self.X),
                self.y,
                lam=lam,
                solver=iterative_solvers.LSQR,
                preconditioner=forward,
            )

            # check that the underlying method converged.
            self.assertTrue(
                exit_status_lsmr["success"],
                f"LSMR solver reported failure to solve system with lam: {lam}.",
            )

            # check that the underlying method converged.
            self.assertTrue(
                exit_status_lsqr["success"],
                f"LSQR solver reported failure to solve system with lam: {lam}.",
            )
            # check that the solutions are the same.
            self.assertTrue(
                np.allclose(w_ne, w_lsmr, rtol=1e-5, atol=1e-5),
                f"The solution to the normal equations differs from the LSMR solution with lam: {lam}.",
            )
            # check that the solutions are the same.
            self.assertTrue(
                np.allclose(w_ne, w_lsqr, rtol=1e-5, atol=1e-5),
                f"The solution to the normal equations differs from the LSQR solution with lam: {lam}.",
            )

    def test_lstsq_iterative_solve_nn(self):
        """Test solving the convex neural network least squares problem with preconditioning."""
        precon = preconditioners.column_norm(self.X, D=self.D)
        expanded_X = operators.expanded_data_matrix(self.X, self.D)
        linear_op = self.network.data_operator(self.X)

        for lam in [1.0, 10.0, 100.0]:
            # solve least-squares problem for optimal weights.
            X_tilde = np.divide(expanded_X, np.sqrt(np.sum(expanded_X ** 2, axis=0)))
            w_ne = direct_solvers.solve_ne(X_tilde, self.y, lam=lam)
            w_ne = precon.matvec(w_ne)

            # try iterative solvers
            w_lsmr, exit_status_lsmr = iterative_solvers.lstsq_iterative_solve(
                linear_op,
                self.y,
                lam=lam,
                solver=iterative_solvers.LSMR,
                preconditioner=precon,
            )

            w_lsqr, exit_status_lsqr = iterative_solvers.lstsq_iterative_solve(
                linear_op,
                self.y,
                lam=lam,
                solver=iterative_solvers.LSQR,
                preconditioner=precon,
            )

            # check that the underlying method converged.
            self.assertTrue(
                exit_status_lsmr["success"],
                f"LSMR solver reported failure to solve system with lam: {lam}.",
            )

            # check that the underlying method converged.
            self.assertTrue(
                exit_status_lsqr["success"],
                f"LSQR solver reported failure to solve system with lam: {lam}.",
            )
            # check that the solutions are the same.
            self.assertTrue(
                np.allclose(w_ne, w_lsmr, rtol=1e-5, atol=1e-5),
                f"The solution to the normal equations differs from the LSMR solution with lam: {lam}.",
            )
            # check that the solutions are the same.
            self.assertTrue(
                np.allclose(w_ne, w_lsqr, rtol=1e-5, atol=1e-5),
                f"The solution to the normal equations differs from the LSQR solution with lam: {lam}.",
            )


class TestLinearSolvers(unittest.TestCase):
    """Test iterative solvers for linear systems."""

    d: int = 50
    n: int = 200

    num_blocks: int = 5
    block_d: int = 10
    block_n: int = 50

    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        # generate PD systems to solve.

        X = self.rng.standard_normal((self.n, self.d))
        self.w_opt = self.rng.standard_normal(self.d)

        self.matrix = np.matmul(X.T, X)
        self.linear_op = aslinearoperator(self.matrix)
        self.targets = self.linear_op.matvec(self.w_opt)

        self.forward = preconditioners.column_norm(self.matrix)

    def test_linear_solver(self):
        """Test solving linear systems."""

        w_iter, exit_status = iterative_solvers.linear_iterative_solve(
            self.linear_op, self.targets
        )

        self.assertTrue(exit_status["success"], "The linear solver reported failure!")

        # solve using numpy.solve

        w_np = np.linalg.solve(self.matrix, self.targets)

        self.assertTrue(
            np.allclose(w_iter, self.w_opt, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match ground truth.",
        )
        self.assertTrue(
            np.allclose(w_iter, w_np, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match direct solver.",
        )

    def test_precon_linear_solver(self):
        """Test solving linear systems with preconditioning."""

        w_iter, exit_status = iterative_solvers.linear_iterative_solve(
            self.linear_op, self.targets, preconditioner=self.forward
        )

        self.assertTrue(exit_status["success"], "The linear solver reported failure!")

        # solve using numpy.solve

        w_np = np.linalg.solve(self.matrix, self.targets)

        self.assertTrue(
            np.allclose(w_iter, self.w_opt, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match ground truth.",
        )
        self.assertTrue(
            np.allclose(w_iter, w_np, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match direct solver.",
        )

    def test_solving_block_systems(self):
        """Test solving block-diagonal linear systems."""

        matrix_blocks = []
        w_opt = []
        targets = []

        for i in range(self.num_blocks):
            block_X = self.rng.standard_normal((self.block_n, self.block_d))
            block_XX = np.matmul(block_X.T, block_X)
            block_opt = self.rng.standard_normal(self.block_d)
            targets.append(np.dot(block_XX, block_opt))

            matrix_blocks.append(block_XX)
            w_opt.append(block_opt)

        w_opt = np.array(w_opt)
        targets = np.array(targets)

        def block_matvec(v):
            results = []
            v = v.reshape(self.num_blocks, self.block_d)
            for i, B in enumerate(matrix_blocks):
                results.append(np.dot(B, v[i]))
            return np.array(results).reshape(-1)

        dp = self.block_d * self.num_blocks
        linear_op = LinearOperator((dp, dp), matvec=block_matvec)

        w_iter, exit_status = iterative_solvers.linear_iterative_solve(
            linear_op, targets
        )

        self.assertTrue(exit_status["success"], "The linear solver reported failure!")

        # compare against ground truth.

        self.assertTrue(
            np.allclose(w_iter, w_opt, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match ground truth.",
        )


if __name__ == "__main__":
    unittest.main()
