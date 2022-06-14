"""
Tests for gradient descent.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from scnn.private.methods.core import gradient_descent as gd
from scnn.private.methods.line_search import backtrack, conditions
from scnn.private.utils import solve_ne
from scnn.private.models import LinearRegression
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestGradientDescent(unittest.TestCase):
    """Test implementation of gradient descent with and without line-search."""

    # basic parameters
    d: int = 50
    n: int = 200
    rng: np.random.Generator = np.random.default_rng(seed=778)

    # line-search parameters
    rho: float = 0.1
    beta: float = 0.8
    init_step_size: float = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        # generate random dataset
        train_set, _, self.wopt = gen_regression_data(
            self.rng,
            self.n,
            0,
            self.d,
            sigma=0.1,
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.wopt = lab.tensor(self.wopt)
        self.y = lab.squeeze(self.y)

        # compute approximation to maximum eigenvalue of the Hessian
        self.L = lab.sum(self.X ** 2) / len(self.y)

        # initialize model
        self.lr = LinearRegression(self.d)
        self.lr.weights = lab.tensor(
            self.rng.standard_normal(self.d, dtype=self.dtype)
        )

        self.backtrack_fn = backtrack.MultiplicativeBacktracker(beta=self.beta)
        self.ls_cond = conditions.Armijo(rho=self.rho)
        self.obj_fn, self.grad_fn = self.lr.get_closures(self.X, self.y)

    def gd_update(self):
        grad = self.lr.grad(self.X, self.y)
        w_plus = gd.gradient_step(
            self.lr.weights,
            grad,
            1.0 / self.L,
        )
        self.lr.weights = w_plus
        return grad

    def ls_update(self):
        grad = self.grad_fn(self.lr.weights)
        self.lr.weights, f1, step_size, exit_state = gd.gd_ls(
            self.lr.weights,
            self.obj_fn(self.lr.weights),
            grad,
            grad,
            self.obj_fn,
            self.grad_fn,
            self.init_step_size,
            self.ls_cond,
            self.backtrack_fn,
        )

        return f1, step_size

    def test_gradient_step(self):
        """Test the basic gradient descent step for guaranteed properties."""

        # gradient descent should minimize a isotropic quadratic in one step:
        # define starting point and isotropic quadratic.
        w_0 = lab.tensor(self.rng.standard_normal(10, dtype=self.dtype))
        center = lab.tensor(self.rng.standard_normal(10, dtype=self.dtype))
        scaling = lab.tensor(self.rng.standard_normal(1, dtype=self.dtype))

        grad = scaling * (w_0 - center)  # compute gradient of quadratic
        w_plus = gd.gradient_step(w_0, grad, 1.0 / scaling)
        self.assertTrue(
            lab.allclose(w_plus, center),
            "Step failed to exactly minimize isotropic quadratic.",
        )

        # gradient descent should satisfy a descent condition
        f0 = self.lr.objective(self.X, self.y)
        grad = self.gd_update()
        f1 = self.lr.objective(self.X, self.y)
        decrease = f1 - f0
        self.assertTrue(
            decrease <= -lab.sum(grad ** 2) / (2 * self.L),
            "Step failed to satisfy the descent lemma.",
        )

        # with enough iterations, GD converges to the true model.
        for i in range(10000):
            self.gd_update()

        opt = solve_ne(self.X, self.y)

        self.assertTrue(
            lab.allclose(
                self.lr.weights,
                opt,
            ),
            "Gradient descent failed to minimize the function!",
        )

    def test_gd_ls(self):
        """Test gradient descent step with Armijo line-search."""
        # the final step-size should satisfy the Armijo condition.
        f0 = self.lr.objective(self.X, self.y)
        grad = self.lr.grad(self.X, self.y)
        w_0 = self.lr.weights
        f1, step_size = self.ls_update()
        w_1 = self.lr.weights

        self.assertTrue(
            self.ls_cond(f0, f1, w_1 - w_0, grad, 0),
            "The final step didn't satisfy the line-search condition.",
        )

        # with enough iterates, the gd w/ line-search should converge to the true model.
        for i in range(1000):
            self.ls_update()

        opt = solve_ne(self.X, self.y)

        self.assertTrue(
            lab.allclose(self.lr.weights, opt),
            "Gradient descent with line-search failed to minimize the function!",
        )


if __name__ == "__main__":
    unittest.main()
