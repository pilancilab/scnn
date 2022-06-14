"""Interface for models.

TODO:
    - A better implementation for batching.
    - A better implementation for getting weights.
    - A better implementation for computing the objective scaling.
    - Rename `weights` to `parameters`.
    - Refactor `weights` to be a list of arrays, instead of a single array.
    """

from math import ceil
from typing import Optional, Tuple, Callable, Union, List, Dict

from scipy.sparse.linalg import (  # type: ignore
    LinearOperator,
    aslinearoperator,
)

import lab

from scnn.private.models.regularizers import Regularizer


class Model:

    """Interface for prediction models.

    Models classes wrap the functionality of a "forward pass", objective
    function, and "backward pass". Models must be able to

        1. Make predictions given data;

        2. Compute the objective associated with their predictions;

        3. Compute first-order derivatives of that objective.

    This abandons the compositionality of auto-differentiation software, but
    allows for optimized computation of the objective and gradient. A model
    has an optional regularizer, which computes some penalty on the model
    parameters.

    Attributes:
        d: the input dimension.
        c: the output dimension.
        weights: a tensor of parameters for the model.
        regularizer: an optional regularizer.
    """

    # public fields

    p: int = 0  # number of hidden units
    c: int = 1  # output dimension
    weights: lab.Tensor  # model parameters
    regularizer: Optional[Regularizer]  # penalty/regularizer

    # TODO: cleanup
    activation_history: Optional[lab.Tensor] = None
    weight_history: Optional[lab.Tensor] = None
    D: Optional[lab.Tensor] = None

    # private fields

    _train: bool = True

    def __init__(self, regularizer: Optional[Regularizer] = None):
        self.regularizer = regularizer

    def _scaling(self, y: lab.Tensor, scaling: Optional[float] = None):
        return scaling * self.c if scaling is not None else len(y)

    def batch_X(
        self, batch_size: Optional[int], X: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:

        if batch_size is None:
            return [{"X": X}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {"X": X[i * batch_size : (i + 1) * batch_size]}
            for i in range(n_batches)
        ]

    def batch_Xy(
        self, batch_size: Optional[int], X: lab.Tensor, y: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:

        if batch_size is None:
            return [{"X": X, "y": y}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "y": y[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def set_weights(self, weights: lab.Tensor):
        self.weights = weights

    def get_weights(self) -> List[lab.Tensor]:
        """Get model weights in an interpretable format.

        :returns: list of tensors.
        """
        return self.weights

    def _forward(self, X: lab.Tensor, w: lab.Tensor, **kwargs) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        :returns: the predictions f(x).
        """

        raise NotImplementedError(
            "A model must be associated with a objective function."
        )

    def _weights(self, w: Optional[lab.Tensor]) -> lab.Tensor:
        return self.weights if w is None else w

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> Union[float, lab.Tensor]:
        """Compute objective associated with examples X and targets y.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X,y)).
        """
        raise NotImplementedError(
            "A model must be associated with a objective function."
        )

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        raise NotImplementedError(
            "A model must be able to compute its gradient with respect to the objective."
        )

    def __call__(
        self,
        X: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,d) array containing the data examples.
        :param w: (optional) specific parameter at which to compute the forward pass.
            Defaults to 'None', in which case the current model state is used.
        :param batch_size: the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :returns: the predictions f(x).
        """
        return lab.concatenate(
            [
                self._forward(**batch, w=self._weights(w), **kwargs)
                for batch in self.batch_X(batch_size, X)
            ],
            axis=0,
        )

    def objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        ignore_regularizer: bool = False,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[float, lab.Tensor]:
        """Compute objective associated with examples X and targets y.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: (optional) specific parameter at which to compute the objective.
            Defaults to 'None', in which case the current model state is used.
        :param ignore_regularizer: (optional) whether or not to ignore the regularizer and return
            *only* the un-regularized objective.
        :param batch_size: the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :returns: objective L(f, (X,y)).
        """
        w = self._weights(w)

        obj = 0.0
        for batch in self.batch_Xy(batch_size, X, y):
            obj += self._objective(**batch, w=w, scaling=y.shape[0], **kwargs)

        if self.regularizer is not None and not ignore_regularizer:
            obj += self.regularizer.penalty(w)

        return obj

    def grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        ignore_regularizer: bool = False,
        return_model_grad: bool = False,
        batch_size: Optional[int] = None,
        step_size: Optional[float] = 1.0,
        **kwargs,
    ) -> Union[lab.Tensor, Tuple[lab.Tensor, lab.Tensor]]:
        """Compute the gradient of the objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: (optional) specific parameter at which to compute the gradient.
            Defaults to 'None', in which case the current model state is used.
        :param ignore_regularizer: (optional) whether or not to ignore the regularizer and return
            *only* the gradient of the un-regularized objective.
        :param batch_size: the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :param step_size: (optional) step-size to use when computing the proximal gradient mapping.
            Defaults to 1.0.
        :returns: the gradient
        """
        w = self._weights(w)
        model_grad = lab.zeros_like(w)
        for batch in self.batch_Xy(batch_size, X, y):
            model_grad += self._grad(
                **batch, w=w, scaling=y.shape[0], **kwargs
            )

        grad = model_grad
        if self.regularizer is not None and not ignore_regularizer:
            grad += self.regularizer.grad(w, model_grad, step_size=step_size)

        if return_model_grad:
            return grad, model_grad

        return grad

    def get_closures(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        ignore_regularizer: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[Callable, Callable]:
        """Returns closures for computing the objective, gradient, and Hessian given (X, y).
            Warning: this closure will retain references to X, y and so can prevent garbage collection of
            these objects.
        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param ignore_regularizer: (optional) whether or not to ignore the regularizer and return
            closures for *only* the un-regularized objective.
        :returns: (objective_fn, grad_fn, hessian_fn).
        """

        def objective_fn(
            w: Optional[lab.Tensor] = None,
            ignore_regularizer=ignore_regularizer,
            batch_size=batch_size,
            **kwargs,
        ):
            return self.objective(
                X,
                y,
                w=w,
                ignore_regularizer=ignore_regularizer,
                batch_size=batch_size,
                **kwargs,
            )

        def grad_fn(
            w: Optional[lab.Tensor] = None,
            ignore_regularizer=ignore_regularizer,
            batch_size=batch_size,
            **kwargs,
        ):
            return self.grad(
                X,
                y,
                w=w,
                ignore_regularizer=ignore_regularizer,
                batch_size=batch_size,
                **kwargs,
            )

        return objective_fn, grad_fn

    def data_operator(self, X: lab.Tensor, **kargs) -> LinearOperator:
        """Construct a matrix operator to evaluate the data operator X v. where
        'v' is an input vector with shape (d,).

        :param X: (n,d) array containing the data examples.
        :returns: LinearOperator which computes the desired product.
        """
        return aslinearoperator(X)
