"""
Pre- and post-processing functions to be run before and after optimization, respectively.
"""

from typing import Optional

import lab

from scnn.private.models import Model


class ProcessingFunction:

    """Abstract class for pre- and post-processing functions."""

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:
        """Prepare/update/refine a model before or after optimization
        begins/ends, optionally using the training set (X, y).

        :param model: the model instance to process.
        :param X: the training features.
        :param y: the training targets.
        :returns: processing_fn(model) --- the updated model.
        """
        raise NotImplementedError(
            "A pre- or post-processing function must implement `__call__`."
        )
