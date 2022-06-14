"""Callback functions to be executed after each iteration of optimization."""
from typing import Tuple, Dict, Optional, Callable
import logging

import numpy as np

import lab

from scnn.private.models import (
    Model,
    Regularizer,
    QuadraticDecomposition,
    AL_MLP,
)
from scnn.private.methods.optimizers.pgd import PGDLS
from scnn.private.methods.optimization_procedures.iterative import (
    IterativeOptimizationProcedure,
)
from scnn.private.methods.optimizers import FISTA
from scnn.private.methods.termination_criteria import GradientNorm
from scnn.private.methods.line_search import (
    QuadraticBound,
    MultiplicativeBacktracker,
    Lassplore,
)
from scnn.private.prox import ProximalOperator


class ObservedSignPatterns:

    """Updates the set of active hyperplane arrangements by simultaneously
    running (S)GD on a ReLU MLP of fixed size."""

    def __init__(self):
        """
        :param relu_mlp: a ReLU MLP to be optimized concurrently with the convex model.
        :param optimizer: an optimizer for the ReLU MLP.
        """

        self.observed_patterns = {}
        self.hasher = hash

    def _get_hashes(self, D: lab.Tensor):
        return np.apply_along_axis(
            lambda z: self.hasher(np.array2string(z)), axis=0, arr=lab.to_np(D)
        )

    def _check_and_store_pattern(self, pattern) -> bool:
        if pattern in self.observed_patterns:
            return False
        else:
            self.observed_patterns[pattern] = True
            return True

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:

        # compute new sign patterns
        patterns, weights = model.sign_patterns(X, y)

        indices_to_keep = lab.tensor(
            [
                self._check_and_store_pattern(pattern)
                for pattern in self._get_hashes(patterns)
            ]
        )
        new_patterns = patterns[:, indices_to_keep]
        new_weights = weights[indices_to_keep]

        if model.activation_history is None or not hasattr(
            model, "activation_history"
        ):
            model.activation_history = new_patterns
            model.weight_history = new_weights
        else:
            model.activation_history = lab.concatenate(
                [model.activation_history, new_patterns], axis=1
            )
            model.weight_history = lab.concatenate(
                [model.weight_history, new_weights], axis=0
            )

        return model


class ConeDecomposition:

    """Convert a gated ReLU model into a ReLU model by solving the cone
    decomposition problem."""

    def __init__(self, solver: Callable):
        """
        :param solver: solver to use when computing the cone decomposition.
        """

        self.solver = solver

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:

        decomposed_model, _ = self.solver(model, X, y)

        return decomposed_model


class ApproximateConeDecomposition:

    """Convert a gated ReLU model into a ReLU model by approximating the cone
    decomposition problem."""

    def __init__(
        self,
        regularizer: Regularizer,
        prox: ProximalOperator,
        max_iters: int = 1000,
        tol: float = 1e-6,
        combined: bool = True,
    ):
        """
        Args:
            regularizer: the regularizer to use in the approximation.
            prox: the proximal operator to use for the given regularizer.
        """

        self.regularizer = regularizer
        self.prox = prox
        self.combined = combined

        # create optimizer to use
        self.optimizer = FISTA(
            1.0,
            QuadraticBound(),
            MultiplicativeBacktracker(beta=0.8),
            Lassplore(alpha=1.25, threshold=5.0),
            prox=self.prox,
        )

        # construct optimization routine
        self.opt_proc = IterativeOptimizationProcedure(
            self.optimizer,
            max_iters=max_iters,
            term_criterion=GradientNorm(tol),
            name="quadratic_decomp",
        )

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:

        # create decomposition to solve
        decomposition = QuadraticDecomposition(
            model.d,
            model.D,
            self.regularizer,
            c=model.c,
            combined=self.combined,
        )

        # form decomposition targets
        targets = -decomposition(X, model.weights)

        if self.combined:
            targets = lab.smax(targets, 0)

        # get logger object
        logger = logging.getLogger("scnn")

        # don't change initialization
        initializer = lambda x: x

        _, decomposition, _ = self.opt_proc(
            logger,
            decomposition,
            initializer,
            (X, targets),
            (X, targets),  # no test data, so pass train data.
            (["objective", "grad_norm"], [], []),
        )

        # compute decomposed weights
        w = decomposition.weights
        v = model.weights + w

        # extract ReLU network
        relu_model = AL_MLP(
            model.d,
            model.D,
            model.U,
            model.kernel,
            regularizer=model.regularizer,
            c=model.c,
        )
        relu_model.weights = lab.stack([v, w])

        return relu_model


class ProximalCleanup:
    """Cleanup the solution to an optimization problem by taking one proximal-
    gradient step."""

    def __init__(self, prox: ProximalOperator):
        """
        :param prox: the proximal-operator to use when cleaning-up the model solution.
        """
        self.prox = prox
        # create optimizer to use
        self.optimizer = PGDLS(
            1.0,
            QuadraticBound(),
            MultiplicativeBacktracker(beta=0.8),
            Lassplore(alpha=1.25, threshold=5.0),
            prox=self.prox,
        )

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:
        cleaned_model, _, exit_state = self.optimizer.step(model, X, y)

        return cleaned_model
