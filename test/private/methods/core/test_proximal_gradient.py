"""
Tests for proximal gradient descent.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from scnn.private.methods.line_search import backtrack, conditions
from scnn.private.models import LinearRegression

from scnn.private.methods.core import proximal_gradient as pgd
from scnn.private.methods.core import gradient_descent as gd
from scnn.private.prox import Identity, L1
from scnn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestProximalGradientDescent(unittest.TestCase):
    """Test implementation of proximal gradient descent with and without line-search."""

    # basic parameters
    d: int = 10
    n: int = 200
    rng: np.random.Generator = np.random.default_rng(seed=778)

    # proximal-gd parameters
    max_iters: int = 1000

    # line-search parameters
    beta: float = 0.8
    init_step_size: float = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)
        # generate random dataset
        (train_set, _, self.wopt) = gen_regression_data(778, self.n, 0, self.d)
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)
        self.wopt = lab.tensor(self.wopt)

        # compute upper-bound on maximum eigenvalue of the Hessian
        self.L = lab.sum(self.X ** 2)

        # initialize model
        self.lr = LinearRegression(self.d)
        self.lr.weights = lab.tensor(self.rng.standard_normal(self.d, dtype=self.dtype))

        self.backtrack_fn = backtrack.MultiplicativeBacktracker(beta=self.beta)
        self.ls_cond = conditions.QuadraticBound()
        self.obj_fn, self.grad_fn = self.lr.get_closures(self.X, self.y)

    def minimize_fn(self, lam):
        prox = L1(lam=lam)
        for i in range(self.max_iters):
            grad = self.lr.grad(self.X, self.y)
            self.lr.weights = pgd.proximal_gradient_step(
                self.lr.weights, grad, 1.0 / self.L, prox
            )

    def minimize_fn_ls(self, lam):
        prox = L1(lam=lam)

        step_size = self.init_step_size
        for i in range(self.max_iters):
            # search proximal path.
            grad = self.grad_fn(self.lr.weights)

            self.lr.weights, f1, step_size, exit_state = pgd.proximal_gradient_ls(
                self.lr.weights,
                self.obj_fn(self.lr.weights),
                grad,
                grad,
                self.obj_fn,
                self.grad_fn,
                step_size,
                self.ls_cond,
                self.backtrack_fn,
                prox,
            )

    def test_proximal_gradient_step(self):
        """Test the basic proximal gradient step for guaranteed properties."""

        # proximal-gradient is the same as gradient descent when the prox is the identity map.
        grad = self.lr.grad(self.X, self.y)
        prox_w_plus = pgd.proximal_gradient_step(
            self.lr.weights, grad, 1.0 / self.L, Identity()
        )

        gd_w_plus = gd.gradient_step(
            self.lr.weights,
            grad,
            1.0 / self.L,
        )

        self.assertTrue(
            lab.allclose(prox_w_plus, gd_w_plus),
            "Proximal-gradient failed to match gradient descent when the prox is the identity operator.",
        )

        # l1-regularized regression with lambda >> 0 should exactly zero all weights when fit with proximal-gd.
        self.minimize_fn(lam=1000)
        self.assertTrue(
            lab.all(self.lr.weights == 0),
            "Proximal GD failed to zero weights when lambda >> 0.",
        )

        # l1-regularized regression with moderate lambda should exactly zero some of the weights.
        self.lr.weights = lab.tensor(self.rng.standard_normal(self.d, dtype=self.dtype))
        self.minimize_fn(lam=5)

        self.assertTrue(
            lab.any(self.lr.weights == 0),
            "Proximal GD failed to zero any weights for moderate lambda.",
        )

    def test_proximal_gradient_ls(self):
        """Test proximal gradient descent with line-search."""

        # the final step-size should satisfy the condition.
        f0 = self.lr.objective(self.X, self.y)
        grad = self.lr.grad(self.X, self.y)
        w_0 = self.lr.weights

        # search proximal path.
        grad = self.grad_fn(self.lr.weights)
        w_1, f1, step_size, exit_state = pgd.proximal_gradient_ls(
            self.lr.weights,
            f0,
            grad,
            grad,
            self.obj_fn,
            self.grad_fn,
            self.init_step_size,
            self.ls_cond,
            self.backtrack_fn,
            L1(lam=5),
        )

        self.assertTrue(exit_state["success"], "The line-search failed.")

        self.assertTrue(
            self.ls_cond(f0, f1, w_1 - w_0, grad, step_size),
            "The final step didn't satisfy the line-search condition when searching prox path.",
        )

        # with enough iterates, proximal gd w/ line-search should converge.
        # check for zeroing with very large lambda.
        self.lr.weights = lab.tensor(self.rng.standard_normal(self.d, dtype=self.dtype))
        self.minimize_fn_ls(lam=100)

        self.assertTrue(
            lab.all(self.lr.weights == 0),
            "Proximal gradient with line-search failed to zero all elements with lambda >> 0.",
        )

        #  check for recovering l2 regression when lambda = 0.
        self.lr.weights = lab.zeros(self.d)
        self.minimize_fn_ls(lam=0.0)

        self.assertTrue(
            lab.allclose(self.lr.objective(self.X, self.y), lab.tensor([0.0])),
            "Proximal gradient with line-search failed to find optimal model when lambda = 0.",
        )

        #  check for zeroing some elements for moderate lambda.
        self.lr.weights = lab.tensor(self.rng.standard_normal(self.d, dtype=self.dtype))
        self.minimize_fn_ls(lam=5)

        self.assertTrue(
            lab.any(self.lr.weights == 0),
            "Proximal gradient with line-search failed to exactly zero some weights for moderate lambda.",
        )


if __name__ == "__main__":
    unittest.main()
