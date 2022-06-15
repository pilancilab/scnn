"""Implementation of augmented Lagrangian method for fitting two-layer MLPs
with orthant constraints."""
from typing import Dict, Any, Tuple, Optional
import lab

from scnn.private.models.model import Model
from scnn.private.models import AL_MLP
from scnn.private.methods.optimizers import Optimizer
from scnn.private.methods.optimizers import MetaOptimizer
from scnn.private.methods.termination_criteria import TerminationCriterion
from scnn.private.methods.core import update_multipliers


class AugmentedLagrangian(MetaOptimizer):

    """Augmented Lagrangian method."""

    def __init__(
        self,
        use_delta_init: bool = True,
        subprob_tol: float = 1e-6,
        omega: float = 1e-3,
        eta_upper: float = 1e-2,
        eta_lower: float = 1e-3,
        tau: float = 2,
    ):
        """
        :param use_delta_init: whether or not to initialize delta using a heuristic rule that seeks to put the penalty
            on the same order as the initial function values.
        :param use_dynamic_tols: whether or not to update the tolerances dynamically during optimization.
            If true, the algorithm uses a Lancelot-like scheme to update delta, eta, and omega.
        :param init_scaling: multiplicate scaling for the initial delta value. Only matters when using the heuristic
            rule to initialize delta.
        :param omega: the starting tolerance to which each sub-problem will be solved. Will be updated dynamically
            during execution when using dynamic tolerances.
        :param eta: the starting tolerance for violation of the constraints. Only matters when using dynamic
            tolerances and will be changed dynamically during optimization.
        :param tau: the multiplicative increase for the penalty parameter when insufficient progress is made
            on the constraints. Only matters when using dynamic tolerances.
        :param c: the power which controls how the constraint tolerance evolves after successful iterations.
            Only matters when using dynamic tolerances.
        :param min_omega: the minimum value of the sub-problem optimization tolerance.
        """

        self.use_delta_init = use_delta_init

        self.subprob_tol = subprob_tol
        self.omega = omega
        self.eta_lower = eta_lower
        self.eta_upper = eta_upper
        self.tau = tau

        self.delta: Optional[float] = None

    def reset(self):
        """Reset the optimizer to its initial state."""

        self.delta = None

    def step(
        self,
        model: AL_MLP,
        inner_term_criterion: TerminationCriterion,
        inner_optimizer: Optimizer,
        X: lab.Tensor,
        y: lab.Tensor,
        batch_index: Optional[int] = None,
    ) -> Tuple[Model, TerminationCriterion, Optimizer, Dict[str, Any]]:
        """Execute one step of the augmented Lagrangian method. Warning: this
        will solve a another optimization problem as sub-routine, which can be
        costly.

        :param model: the model to update.
        :param inner_term_criterion: the termination criterion of the inner optimization
            routine to which this meta optimizer is being applied.
        :param inner_optimizer: the optimizer for the inner optimization
            routine to which this meta optimizer is being applied.
        :param X: the (n,d) data matrix to use in the update.
        :param y: the (n,) vector of targets.
        :param batch_index: NOT USED. The index of the current mini-batch.
        :returns: the updated model and exit status of any sub-procedures (ie. line-search)
        """

        # initial setup; no dual updates.
        if self.delta is None:
            self.delta = model.delta

            if self.use_delta_init:
                inner_term_criterion.tol = self.omega
                self.initializing_delta = True
            else:
                inner_term_criterion.tol = self.subprob_tol
                self.initializing_delta = False
        else:
            e_gap, i_gap = model.constraint_gaps(X)
            gap_norm = lab.sum(e_gap**2) + lab.sum(i_gap**2)

            # TODO: sometimes the lower window leads to weird failure cases.
            if self.initializing_delta:

                # the previous sub-problem did not make enough progress on the constraints;
                # increase penalty strength
                if gap_norm > self.eta_upper:
                    self.delta = self.delta * self.tau
                    model.delta = self.delta
                # the previous sub-problem made too much progress on the constraints;
                # decrease penalty strength
                elif gap_norm < self.eta_lower:
                    self.delta = self.delta / self.tau
                    model.delta = self.delta
                    inner_term_criterion.tol = (
                        inner_term_criterion.tol / self.tau
                    )
                else:
                    # gap within starting "window".
                    self.initializing_delta = False
                    inner_term_criterion.tol = self.subprob_tol

            else:
                # update the dual parameters.
                model.e_multipliers, model.i_multipliers = update_multipliers(
                    model.e_multipliers,
                    model.i_multipliers,
                    e_gap,
                    i_gap,
                    model.delta,
                )

                # mechanism for tightening solution tolerances.
                if self.tau * gap_norm < inner_term_criterion.tol:
                    # reduce tolerance to ensure continued progress.
                    inner_term_criterion.tol = (
                        inner_term_criterion.tol / self.tau
                    )

        inner_optimizer.reset()
        exit_status: Dict[str, Any] = {}

        return model, inner_term_criterion, inner_optimizer, exit_status
