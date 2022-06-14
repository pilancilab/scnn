"""
Tests for line-search conditions.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab
from scnn.private.methods.line_search import conditions


@parameterized_class(lab.TEST_GRID)
class TestLineSearchConditions(unittest.TestCase):
    """Tests for line-search conditions."""

    rng: np.random.Generator = np.random.default_rng(778)

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

    def test_armijo(self):
        """Test regular Armijo (sufficient progress) condition."""

        armijo = conditions.Armijo(rho=0.5)
        step_size = 1

        f0 = f1 = 0
        grad = lab.ones((5))

        # Armijo fails when no progress is made.
        self.assertFalse(
            armijo(f0, f1, -grad, grad, step_size),
            "The line-search succeeded despite making no progress.",
        )

        # Armijo succeeds when exactly the right progress is made.
        f1 = f0 + lab.sum(lab.multiply(grad, -grad)) / 2
        self.assertTrue(
            armijo(f0, f1, -grad, grad, step_size),
            "The line-search failed when exactly the minimum amount of progress was made.",
        )

        # Armijo fails if epsilon less progress is made
        f1 = 1e-4 + f0 - lab.sum(grad ** 2) / 2
        self.assertFalse(
            armijo(f0, f1, -grad, grad, step_size),
            "The line-search succeeded despite making slightly too little progress.",
        )

        # Armijo supports matrix steps and gradients
        grad = lab.ones((5, 5))
        f1 = f0 + lab.sum(lab.multiply(grad, -grad)) / 2
        self.assertTrue(
            armijo(f0, f1, -grad, grad, step_size),
            "The line-search failed with matrix gradients.",
        )

        # Armijo supports rotated search directions
        grad = lab.tensor(self.rng.standard_normal(4, dtype=self.dtype))
        step = -2 * lab.multiply(lab.tensor([0.1, 0.2, 0.1, 0.3]), grad)
        self.assertFalse(
            armijo(f0, f0, step, grad, step_size),
            "The line-search failed with rotated search directions.",
        )
        armijo.rho = 1e-1
        f1 = f0 - 0.5
        self.assertTrue(
            armijo(f0, f1, step, grad, step_size),
            "The line-search failed with rotated search directions.",
        )


if __name__ == "__main__":
    unittest.main()
