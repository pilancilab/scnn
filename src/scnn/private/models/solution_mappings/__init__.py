"""Solution mappings for convex and non-convex formulations of neural network
training problems."""

from .mlps import (
    is_relu_compatible,
    is_grelu_compatible,
    construct_nc_manual,
    construct_nc_torch,
)

__all__ = [
    "is_relu_compatible",
    "is_grelu_compatible",
    "is_compatible",
    "get_nc_mlp",
    "get_nc_formulation",
    "is_compatible",
]

from typing import Union

import torch

from scnn.private.models.model import Model
from scnn.private.models.convex import (
    AL_MLP,
    ConvexMLP,
)
from scnn.private.models.linear import LinearRegression


def is_compatible(torch_model: torch.nn.Module) -> bool:
    """Check to see if there is a solution mapping mapping which is compatible
    with the architecture of the given model.

    :param torch_model: an instance of torch.nn.Module for which we want a convex program.
    :returns: true or false.
    """

    return is_relu_compatible(torch_model) or is_grelu_compatible(torch_model)


def get_nc_formulation(
    convex_model: Model,
    remove_sparse: bool = False,
) -> Union[torch.nn.Module, Model]:

    grelu = True
    if isinstance(convex_model, (AL_MLP)):
        grelu = False

    if isinstance(convex_model, ConvexMLP):
        return construct_nc_manual(convex_model, grelu, remove_sparse)
    elif isinstance(convex_model, LinearRegression):
        return convex_model
    else:
        raise ValueError(
            f"Model {convex_model} not recognized. Please add it to 'solution_mappings.mlps.py'"
        )
