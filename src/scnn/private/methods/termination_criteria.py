"""Termination criteria for optimization methods.

TODO:
    - Update `StepLength` to be the gradient mapping. The major
        difference is multiplication by a step-size.
"""

from typing import Optional

import lab
from scnn.private.models.model import Model


class TerminationCriterion:
    """Base class for termination criteria.

    A boolean function with one or more tuning parameters that returns `True`
    when an optimization procedure should terminate and `False` otherwise.

    Attributes:
        tol: a parameter controlling sensitivity of the termination criterion.
    """

    tol: float

    def __init__(self, tol: float):
        """
        Args:
            tol: a tolerance parameter controlling sensitivity of the
                termination criterion.
        """

        self.tol = tol

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Evaluate the termination criterion given a model and a dataset.

        The current objective value and gradient are optional parameters;
        these should be supplied if they have been pre-calculated for another
        purpose. Otherwise, the criterion will compute them as necessary.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value.
                Provide only if already computed.
            grad: the current gradient of the objective.
                Provide only if already computed.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """
        raise NotImplementedError(
            "Termination criteria must implement __call__!"
        )


class ConstrainedTerminationCriterion(TerminationCriterion):
    """Convergence criterion for constrained optimization problems.

    Base class for termination criteria for problems with constraints.
    Provides functions for computing the constraint violations.

    Attributes:
        obj_tol: the tolerance for the primal objective.
        constraint_tol: the tolerance for violation of the constraints.
    """

    obj_tol: float
    constraint_tol: float

    def __init__(self, obj_tol: float, constraint_tol: float):
        """
        Args:
            obj_tol: the tolerance for determining optimality of the primal
                optimization problem.
            constraint_tol: the tolerance for determining feasibility.
        """
        self.obj_tol = obj_tol
        self.constraint_tol = constraint_tol

    def constraint_violations(self, model: Model, X: lab.Tensor) -> float:
        """Compute current constraint violations.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.

        Returns:
            Squared l2-norm of constraint violations.
        """

        e_gap, i_gap = model.constraint_gaps(X)
        return lab.sum(e_gap ** 2 + lab.smax(i_gap, 0) ** 2)


class GradientNorm(TerminationCriterion):
    """First-order optimality criterion.

    Terminate optimization if and only if the norm of minimum-norm subgradient
    is below a certain tolerance.


    Attributes:
        tol: the tolerance for the gradient norm. The objective is
            approximately stationary if the gradient norm is less than `tol`.
    """

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Terminate if gradient norm is sufficiently small.

        Determine if the norm of the minimum-norm sub-gradient (or gradient if
        the function is smooth) is small enough (according to self.tol) to
        constitute a first-order stationary point.

        The current objective value and gradient are optional parameters;
        these should be supplied if they have been pre-calculated for another
        purpose. Otherwise, the criterion will compute them as necessary.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value. NOT USED.
            grad: the current gradient of the objective.
                Provide only if already computed.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """

        if grad is None:
            grad = model.grad(X, y)

        return lab.sum(grad ** 2) <= self.tol


class StepLength(TerminationCriterion):
    """Criterion based on length of the most recent step.

    Terminate optimization if and only if the norm of the last step was
    below a certain tolerance.

    Notes:
        This criterion can significant increase the memory requirements
        of an optimization procedure due to the `previous_weights` attributes.

    Attributes:
        previous_weights: the parameters of the model from the previous
            iteration. These are necessary to compute the length of the
            step.

    """

    previous_weights: Optional[lab.Tensor] = None

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Terminate if step-length is sufficiently small.

        Determine if the norm of previous step is small enough (according to
        `self.tol`) to indicate the method has converged.

        The current objective value and gradient are not used.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value. NOT USED.
            grad: the current gradient of the objective. NOT USED.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """
        if self.previous_weights is None:
            self.previous_weights = model.weights
            return False

        step = model.weights - self.previous_weights
        self.previous_weights = model.weights

        return lab.sqrt(lab.sum(step ** 2)) <= self.tol


class ConstrainedHeuristic(ConstrainedTerminationCriterion):
    """Terminate if the gradient norm and constraint violations are both small.

    A heuristic condition which terminates optimization if and only if the
    norm of minimum-norm subgradient is below a certain tolerance and the norm
    of the constraints violations is below a separate tolerance.

    Notes:
        - The gradient norm must be computed with respect a penalized
            objective; otherwise, termination is only possible if a solution
            exists in the interior of the constraint set.

        - This criterion is not the same as checking stationarity of the
            Lagrangian, but appears to work well in practice for penalized
            objectives.

    Attributes:
        obj_tol: the tolerance for the gradient norm. The penalized objective
            is approximately stationary if the gradient norm is less than
            `grad_tol`.
        constraint_tol: the tolerance for violation of the constraints.
            The model is approximately feasible if the norm of the constraint
            violations is less than `constraint_tol`.
    """

    obj_tol: float
    constraint_tol: float

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Terminate if gradient norm and constraint violations are small.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value. NOT USED.
            grad: the current gradient of the objective.
                Provide only if already computed.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """
        if grad is None:
            grad = model.grad(X, y)

        gaps = self.constraint_violations(model, X)

        if gaps <= self.constraint_tol:
            return lab.sum(grad ** 2) <= self.obj_tol

        return False


class LagrangianGradNorm(ConstrainedTerminationCriterion):
    """First-order optimality criterion for primal-dual methods.

    Terminate optimization if and only if the norm of minimum-norm subgradient
    of the Lagrangian function is below a certain tolerance and the current
    point is approximately feasible. This criterion is only supported for
    solvers which maintain both primal and dual parameters. The gradient is
    computed with respect the primal parameters.

    Attributes:
        obj_tol: the tolerance for the gradient norm. The Lagrangian
            is approximately stationary if the gradient norm is less than
            `grad_tol`.
        constraint_tol: the tolerance for violation of the constraints.
            The model is approximately feasible if the norm of the constraint
            violations is less than `constraint_tol`.
    """

    obj_tol: float
    constraint_tol: float

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Terminate if the Lagrangian is approximately stationary and the
        constraint violations are small.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value. NOT USED.
            grad: the current gradient of the objective. NOT USED.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """

        gaps = self.constraint_violations(model, X)
        if gaps <= self.constraint_tol:
            return lab.sum(model.lagrangian_grad(X, y) ** 2) <= self.obj_tol

        return False
