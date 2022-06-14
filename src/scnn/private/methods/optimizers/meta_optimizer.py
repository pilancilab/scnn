"""Implementation of augmented Lagrangian method for fitting two-layer MLPs
with orthant constraints."""
from typing import Dict, Any, Tuple, Optional

import lab

from scnn.private.models.model import Model
from scnn.private.methods.optimizers import Optimizer
from scnn.private.methods.termination_criteria import TerminationCriterion


class MetaOptimizer(Optimizer):

    """Meta optimization routine."""

    step_size: float = 0.0

    def step(
        self,
        model: Model,
        inner_term_criterion: TerminationCriterion,
        inner_optimizer: Optimizer,
        X: lab.Tensor,
        y: lab.Tensor,
        batch_index: Optional[int] = None,
    ) -> Tuple[Model, TerminationCriterion, Optimizer, Dict[str, Any]]:
        """Execute one step of a meta optimization routine.

        :param model: the model to update.
        :param inner_term_criterion: the termination criterion of the inner optimization
            routine to which this meta optimizer is being applied.
        :param inner_optimizer: the optimizer for the inner optimization
            routine to which this meta optimizer is being applied.
        :param X: the (n,d) data matrix to use in the update.
        :param y: the (n,) vector of targets.
        :returns: the updated model, inner termination criterion, inner optimizer,
            and exit status of any sub-procedures (ie. line-search)
        """

        raise NotImplementedError("A meta optimizer must implement 'step'!")

    def outer(self, model: Model) -> Model:
        """Put model in inner-loop mode.

        :param model: the model.
        :returns: updated model.
        """

        return model

    def inner(self, model: Model) -> Model:
        """Put model in outer-loop mode.

        :param model: the model.
        :returns: updated model.
        """

        return model
