"""Functions for interfacing between the public API and the models, optimizers,
and data representations used in the :module:`private`."""

from .data import (
    normalized_into_input_space,
    input_into_normalized_space,
    process_data,
)

from .models import (
    build_internal_regularizer,
    build_internal_model,
    update_public_model,
    build_public_model,
)

from .solvers import (
    build_prox_operator,
    build_fista,
    build_optimizer,
)

from .metrics import (
    build_metrics_tuple,
    update_public_metrics,
)

from .utils import (
    get_logger,
    set_device,
)

__all__ = [
    "normalized_into_input_space",
    "input_into_normalized_space",
    "process_data",
    "build_internal_regularizer",
    "build_internal_model",
    "update_public_model",
    "build_public_model",
    "build_prox_operator",
    "build_fista",
    "build_optimizer",
    "build_metrics_tuple",
    "update_public_metrics",
    "get_logger",
    "set_device",
]
