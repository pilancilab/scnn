"""The FISTA accelerated proximal gradient descent method."""
from typing import Optional, cast, Dict, Any, Tuple

import lab

from scnn.private.methods.core.proximal_gradient import (
    fista_ls,
)
from scnn.private.models.model import Model
from scnn.private.methods.optimizers.proximal_optimizer import ProximalLSOptimizer

from scnn.private.prox import ProximalOperator
from scnn.private.methods.line_search import StepSizeUpdater, LSCondition, Backtracker

# constants
F_VALS: str = "function_values"
GM: str = "gradient_mapping"


class FISTA(ProximalLSOptimizer):

    """Fast iterative shrinkage-thresholding method."""

    def __init__(
        self,
        init_step_size: float,
        ls_cond: LSCondition,
        backtrack_fn: Backtracker,
        update_step_size: StepSizeUpdater,
        prox: ProximalOperator,
        mu: float = 0.0,
        restart_rule: Optional[str] = GM,
    ):
        """
        :param step_fn: a function to call to execute one step of the iterative
            method. It should have the same signature as 'methods.gradient_descent.gd_ls'.
        :param init_step_size: first step-size to try when running the line-search.
        :param ls_cond: the line-search condition to check.
        :param backtrack: a rule for calculating the next step-size to try.
        :param update_step_size: a rule for updating the step-size after each line-search.
        :param prox: a proximal operator. See 'proximal_ops'.
        :param mu: (optional) a lower-bound on the strong-convexity parameter of the objective.
            The method defaults to the parameter sequence for non-strongly convex functions when
            mu is not supplied.
        :param restart_rule: (optional) rule used to trigger a "restart" of the optimizer.
        """
        super().__init__(
            fista_ls,
            init_step_size,
            ls_cond,
            backtrack_fn,
            update_step_size,
            prox,
        )

        self.mu = mu
        self.v: Optional[lab.Tensor] = None
        self.t: float = 1.0
        self.restart_rule = restart_rule

        self.disp = None

    def reset(self):
        """Reset the secondary sequences to their initial state."""
        super().reset()
        # reset acceleration specific memory.
        self.restart()

    def restart(self):
        self.t = 1.0
        self.v = None

    def _check_restart(
        self, weights: lab.Tensor, old_weights: lab.Tensor, old_v: lab.Tensor
    ):
        if self.restart_rule is None or self.f0 is None or self.f1 is None:
            return
        # restart when function values increase
        elif self.restart_rule == F_VALS and self.f0 < self.f1:
            self.restart()
        # restart when step makes an obtuse angle with the gradient mapping.
        elif self.restart_rule == GM:
            if lab.sum((weights - old_weights) * (old_v - weights)) > 0.0:
                self.restart()

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
        # compute objective and gradient without the non-smooth regularizer.
        obj_fn, grad_fn = model.get_closures(
            X, y, ignore_regularizer=True, batch_size=batch_size
        )

        if self.v is None:
            self.v = model.weights

        grad = grad_fn(self.v)
        descent_dir = grad

        # fv: function value at extrapolation.
        self.fv = obj_fn(self.v)
        # f0: function value at previous parameter
        self.f0 = self.f1
        old_weights = model.weights
        old_v = self.v

        # update model
        (
            model.weights,
            self.v,
            self.t,
            self.f1,
            self.step_size,
            exit_state,
        ) = self.step_fn(
            model.weights,
            self.fv,
            descent_dir,
            grad,
            obj_fn,
            grad_fn,
            self.step_size,
            self.ls_cond,
            self.backtrack_fn,
            self.prox,
            self.v,
            self.t,
            mu=self.mu,
        )

        # update step-size
        self.step_size = self.update_step_size(
            self.step_size,
            self.old_step_size,
            model.weights - old_v,
            self.fv,
            self.f1,
            grad,
        )
        self.old_step_size = self.step_size

        # check if we must restart the accelerated method.
        self._check_restart(model.weights, old_weights, old_v)

        return model, None, exit_state
