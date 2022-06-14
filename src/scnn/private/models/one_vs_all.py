"""One-vs-all-classification model."""

from typing import Optional, Tuple, List

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
from scnn.private.loss_functions import squared_error


class OneVsAllModel(Model):
    """A one-vs-all model."""

    def __init__(
        self,
        d: int,
        per_class_models: List[Model],
        c: int = 1,
        regularizer: Optional[Regularizer] = None,
    ):
        assert len(per_class_models) == c
        super().__init__(regularizer)

        self.per_class_models = per_class_models

        self.d = d
        self.c = c

    @property
    def weights(self):
        weights = lab.stack([model.weights for model in self.per_class_models])
        return weights

    def _forward(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: (NOT USED) parameter at which to compute the forward pass.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :returns: predictions for X.
        """
        return lab.concatenate(
            [model(X) for model in self.per_class_models], axis=1
        )

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute objective associated with examples X and targets y.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: (NOT USED) specific parameter at which to compute the forward pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X, y)).
        """
        return squared_error(self._forward(X, None), y) / (
            2 * self._scaling(y, scaling)
        )

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        flatten: bool = False,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the l2 objective with respect to the model
        weights.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: (NOT USED) parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        grad = lab.stack(
            [
                model.grad(X, y, w=None, flatten=False, scaling=scaling)
                for model in self.per_class_models
            ]
        )

        if flatten:
            grad = grad.reshape(-1)

        return grad
