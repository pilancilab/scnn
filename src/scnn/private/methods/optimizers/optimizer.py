"""Base class for iterative optimizers."""
from typing import Callable, Optional, Tuple, Dict, Any

import lab

from scnn.private.models.model import Model
from scnn.private.methods.line_search import (
    StepSizeUpdater,
    LSCondition,
    Backtracker,
)


class Optimizer:

    """The base class for optimization methods.

    Optimizer is a stateful wrapper around a 'step_fn' --- a function which can
    be called to compute one step of an iterative optimization method.
    """

    def __init__(
        self,
        step_fn: Callable,
        step_size: float,
        update_step_size: StepSizeUpdater,
    ):
        """
        :param step_fn: a function to call to execute one step of the iterative
            method. It should have the same signature as 'methods.gradient_descent.gd'.
        :param step_size: the constant step-size to use.
        :param update_step_size: a rule for updating the step-size after each line-search.
        """
        self.step_fn = step_fn
        self.step_size = self.init_step_size = step_size
        self.old_step_size = step_size

        self.update_step_size = update_step_size

        self.f0: Optional[float] = None
        self.f1: Optional[float] = None

    def reset(self):
        """Reset the optimizer to its initial state."""

        self.step_size = self.init_step_size
        self.f0 = None
        self.f1 = None

    def step(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        f0: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
        batch_index: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Model, Optional[float], Dict[str, Any]]:
        """Execute one step of an iterative optimization method.

        :param model: the model to update.
        :param X: the (n,d) data matrix to use in the update.
        :param y: the (n,) vector of targets.
        :param f0: (optional) current objective value. Can be provided to re-use computations.
        :param grad: (optional) current objective gradient. Can be provided to re-use computations.
        :param batch_index: NOT USED. The index of the current mini-batch.
        :param batch_size: the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :returns: the updated model and exit status of any sub-procedures (ie. line-search)
        """
        # compute gradient and update model.
        if grad is None:
            grad = model.grad(X, y, batch_size=batch_size)

        old_weights = model.weights
        model.weights = self.step_fn(model.weights, grad, self.step_size)

        # update step-size
        self.step_size = self.update_step_size(
            self.step_size,
            self.old_step_size,
            model.weights - old_weights,
            self.f0,
            self.f1,
            grad,
        )

        return model, None, {"step_size": self.step_size}


class LSOptimizer(Optimizer):

    """Optimization method with line-search.

    Like optimizer, this is a stateful wrapper around a 'step_fn', which can be
    called to execute one step of the f.o. method, including the line-search.
    """

    def __init__(
        self,
        step_fn: Callable,
        init_step_size: float,
        ls_cond: LSCondition,
        backtrack_fn: Backtracker,
        update_step_size: StepSizeUpdater,
    ):
        """
        :param step_fn: a function to call to execute one step of the iterative
            method. It should have the same signature as 'methods.gradient_descent.gd_ls'.
        :param init_step_size: first step-size to try when running the line-search.
        :param ls_cond: the line-search condition to check.
        :param backtrack: a rule for calculating the next step-size to try.
        :param update_step_size: a rule for updating the step-size after each line-search.
        """
        super().__init__(step_fn, init_step_size, update_step_size)

        self.ls_cond = ls_cond
        self.backtrack_fn = backtrack_fn

    def reset(self):
        """Reset the optimizer to its initial state."""
        super().reset()
        self.old_step_size = None

    def step(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        f0: Optional[float] = None,
        grad: Optional[lab.Tensor] = None,
        batch_index: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Model, Optional[float], Dict[str, Any]]:
        """Execute one step of an iterative optimization method.

        :param model: the model to update.
        :param X: the (n,d) data matrix to use in the update.
        :param y: the (n,) vector of targets.
        :param f0: (optional) current objective value. Can be provided to re-use computations.
        :param grad: (optional) current objective gradient. Can be provided to re-use computations.
        :param batch_index: NOT USED. The index of the current mini-batch.
        :param batch_size: the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :returns: (model, status_dict) --- the updated model and a dictionary describing the optimizer's exit state.
        """

        obj_fn, grad_fn = model.get_closures(X, y, batch_size=batch_size)

        if grad is None:
            grad = grad_fn(model.weights)

        if self.f1 is None:
            self.f0 = obj_fn(model.weights)
        else:
            self.f0 = self.f1

        old_weights = model.weights
        model.weights, self.f1, self.step_size, exit_state = self.step_fn(
            model.weights,
            self.f0,
            grad,
            grad,
            obj_fn,
            grad_fn,
            self.step_size,
            self.ls_cond,
            self.backtrack_fn,
            prox=None,
        )

        # update step-size
        self.step_size = self.update_step_size(
            self.step_size,
            self.old_step_size,
            model.weights - old_weights,
            self.f0,
            self.f1,
            grad,
        )
        self.old_step_size = self.step_size

        return model, self.f1, exit_state
