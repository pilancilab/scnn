"""Base class for proximal-gradient optimizers."""
from typing import Callable, Optional, Tuple, Any, Dict

import lab

from scnn.private.models.model import Model
from scnn.private.methods.optimizers.optimizer import Optimizer, LSOptimizer
from scnn.private.prox import ProximalOperator
from scnn.private.methods.line_search import (
    StepSizeUpdater,
    LSCondition,
    Backtracker,
)


class ProximalOptimizer(Optimizer):

    """The base class for proximal-gradient-type methods.

    Optimizer is a stateful wrapper around a 'step_fn' --- a function which can
    be called to compute one step of an iterative optimization method.
    """

    def __init__(
        self,
        step_fn: Callable,
        step_size: float,
        prox: ProximalOperator,
        update_step_size: StepSizeUpdater,
    ):

        """
        :param step_fn: a function to call to execute one step of the iterative
            method. It should have the same signature as 'core.proximal_gradient.proximal_gradient_step'.
        :param step_size: the constant step-size to use.
        :param prox: a proximal operator. See 'proximal_ops'.
        :param update_step_size: a rule for updating the step-size after each line-search.
        """
        super().__init__(step_fn, step_size, update_step_size)
        self.prox = prox

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
        :returns: the updated model and empty exit state (no sub-procedures are called).
        """
        # compute gradient without non-smooth regularizer.
        grad = model.grad(X, y, ignore_regularizer=True, batch_size=batch_size)

        old_weights = model.weights
        model.weights = self.step_fn(model.weights, grad, self.step_size, self.prox)

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


class ProximalLSOptimizer(LSOptimizer):

    """First-order optimization method with line-search.

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
        prox: ProximalOperator,
    ):
        """
        :param step_fn: a function to call to execute one step of the iterative
            method. It should have the same signature as 'methods.gradient_descent.gd_ls'.
        :param init_step_size: first step-size to try when running the line-search.
        :param ls_cond: the line-search condition to check.
        :param backtrack: a rule for calculating the next step-size to try.
        :param update_step_size: a rule for updating the step-size after each line-search.
        :param prox: a proximal operator. See 'proximal_ops'.
        """
        super().__init__(
            step_fn, init_step_size, ls_cond, backtrack_fn, update_step_size
        )
        self.prox = prox

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
        :returns: the updated model and exit state of the line-search.
        """
        # model must be wrapped in a regularizer.
        obj_fn, grad_fn = model.get_closures(
            X, y, ignore_regularizer=True, batch_size=batch_size
        )

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
            prox=self.prox,
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

        return model, None, exit_state
