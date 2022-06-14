"""Wrapper for PyTorch Sequential module ensuring that it meets the interface
specific in model.py."""

from typing import Optional, Union, Tuple

import torch

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
import scnn.private.loss_functions as loss_fns


class SequentialWrapper(torch.nn.Sequential, Model):
    """Model-style wrapper for PyTorch Sequential module that provides an
    interface for objective and gradient computation."""

    activation_history: Optional[torch.Tensor] = None
    weight_history: Optional[torch.Tensor] = None

    def __init__(
        self,
        *modules,
        loss_fn=loss_fns.squared_error,
        regularizer: Optional[Regularizer] = None,
    ):

        super().__init__(*modules)

        self.loss_fn = loss_fn
        self.regularizer = regularizer
        self.layers = modules

    @property
    def weights(self):
        return torch.cat([param.ravel() for param in self.parameters()])

    def forward(self, X: torch.Tensor, batch_size=None):
        return super().forward(X)

    def objective(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        ignore_regularizer: bool = False,
        scaling: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute objective associated with examples X and targets y.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: (NOT USED) specific parameter at which to compute the objective.
        :param ignore_regularizer: (optional) whether or not to ignore the regularizer and return
            *only* the un-regularized objective.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :param batch_size: (NOT USED) the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :returns: objective L(f, (X,y)).
        """
        # hard-code squared loss with weight decay

        obj = self.loss_fn(self.__call__(X), y) / self._scaling(y, scaling)

        for layer in self.layers:
            inner_reg = layer.get_regularizer()

            if inner_reg is not None and not ignore_regularizer:
                obj += inner_reg.penalty(layer.parameters())
            elif self.regularizer is not None and not ignore_regularizer:
                obj += self.regularizer.penalty(layer.parameters())

        return obj

    def grad(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        ignore_regularizer: bool = False,
        return_model_grad: bool = False,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[lab.Tensor, Tuple[lab.Tensor, lab.Tensor]]:
        """Compute the gradient of the objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: (NOT USED) specific parameter at which to compute the gradient.
        :param ignore_regularizer: (optional) whether or not to ignore the regularizer and return
            *only* the gradient of the un-regularized objective.
        :param batch_size: (NOT USED) the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :returns: the gradient
        """
        with torch.set_grad_enabled(True):
            # clear current gradients
            self.zero_grad()

            # compute loss and call autodiff engine
            loss = self.objective(X, y, w, ignore_regularizer=ignore_regularizer)
            loss.backward()

        return torch.cat([param.grad.ravel() for param in self.parameters()])

    def sign_patterns(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, lab.Tensor]:
        """Compute the gradient of the l2 objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: (NOT USED) specific parameter at which to compute the sign patterns.
        :returns: the set of sign patterns active at w or the current models parameters if w is None.
        """
        with torch.no_grad():
            first_layer = next(next(next(self.children()).children()).children())

            # The first layer *must* be a linear map.
            assert isinstance(first_layer, torch.nn.Linear)

            return (
                torch.sign(torch.max(first_layer(X), torch.tensor(0))),
                first_layer.weight,
            )
