"""Convert models from :module:`scnn.models` into internal representations
and vice versa."""

from typing import Optional, List, Tuple

import numpy as np

import lab

from scnn.regularizers import (
    Regularizer,
    NeuronGL1,
    FeatureGL1,
    L2,
    L1,
)

from scnn.models import (
    Model,
    LinearModel,
    ConvexGatedReLU,
    NonConvexGatedReLU,
    ConvexReLU,
    NonConvexReLU,
)

from scnn.private.models import (
    ConvexMLP,
    AL_MLP,
    ReLUMLP,
    GatedReLUMLP,
    GroupL1Regularizer,
    FeatureGroupL1Regularizer,
    L2Regularizer,
    L1Regularizer,
    LinearRegression,
)
from scnn.activations import compute_activation_patterns

from scnn.private.models import Model as InternalModel
from scnn.private.models import Regularizer as InternalRegularizer


def build_internal_regularizer(
    regularizer: Optional[Regularizer] = None,
) -> InternalRegularizer:
    """Convert public-facing regularizer objects into private implementations.

    Args:
        regularizer: a regularizer object from the public API.

    Returns:
        An internal regularizer object with the same state as the public regularizer.
    """
    reg: Optional[InternalRegularizer] = None

    lam = 0.0
    if regularizer is not None:
        lam = regularizer.lam

    if isinstance(regularizer, NeuronGL1):
        reg = GroupL1Regularizer(lam)
    elif isinstance(regularizer, FeatureGL1):
        reg = FeatureGroupL1Regularizer(lam)
    elif isinstance(regularizer, L2):
        reg = L2Regularizer(lam)
    elif isinstance(regularizer, L1):
        reg = L1Regularizer(lam)

    return reg


def build_internal_model(
    model: Model, regularizer: Regularizer, X_train: lab.Tensor
) -> InternalModel:
    """Convert public-facing model objects into private implementations.

    Args:
        model: a model object from the public API.
        regularizer: a regularizer object from the public API.
        X_train: the :math:`n \\times d` training set.

    Returns:
        An internal model object with the same state as the public model.
    """
    assert isinstance(model, (LinearModel, ConvexReLU, ConvexGatedReLU))

    internal_model: InternalModel
    d, c = model.d + model.bias, model.c
    internal_reg = build_internal_regularizer(regularizer)

    if isinstance(model, LinearModel):
        return LinearRegression(d, c, regularizer=internal_reg)

    D, G = lab.all_to_tensor(
        compute_activation_patterns(
            lab.to_np(X_train), model.G, bias=model.bias
        )
    )

    if isinstance(model, ConvexReLU):
        internal_model = AL_MLP(
            d,
            D,
            G,
            "einsum",
            delta=1000,
            regularizer=internal_reg,
            c=c,
        )
    elif isinstance(model, ConvexGatedReLU):
        internal_model = ConvexMLP(
            d, D, G, "einsum", regularizer=internal_reg, c=c
        )
    else:
        raise ValueError(f"Model object {model} not supported.")

    return internal_model


def extract_bias(weights: lab.Tensor, bias: bool = False) -> List[np.ndarray]:
    """Extract optional bias columns from model weights.

    Args:
        weights: the weights or parameters of the model.
        bias: whether or not the weights contain a bias term.

    Returns
    """
    weights = lab.to_np(weights)

    if bias:
        return [weights[..., 0:-1], weights[..., -1]]
    else:
        return [weights]


def extract_gates_bias(
    G: lab.Tensor, bias: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    G = lab.to_np(G)
    if bias:
        return (G[0:-1], G[-1])
    else:
        return (G, None)


def update_public_model(model: Model, internal_model: InternalModel) -> Model:
    """Update public-facing model object to match state of internal model.

    Args:
        model: the public-facing model.
        internal_model: the internal model object.

    Returns:
        The updated public-facing model.
    """
    # update neuron count
    model.p = internal_model.p

    if isinstance(model, ConvexGatedReLU):
        assert isinstance(internal_model, ConvexMLP)
        model.set_parameters(extract_bias(internal_model.weights, model.bias))
        model.G, model.G_bias = extract_gates_bias(
            internal_model.U, model.bias
        )
    elif isinstance(model, ConvexReLU):
        assert isinstance(internal_model, AL_MLP)
        model.set_parameters(
            extract_bias(internal_model.weights[0], model.bias)
            + extract_bias(internal_model.weights[1], model.bias)
        )

        model.G, model.G_bias = extract_gates_bias(
            internal_model.U, model.bias
        )
    elif isinstance(model, LinearModel):
        model.set_parameters(extract_bias(internal_model.weights, model.bias))

    return model


def build_public_model(
    internal_model: InternalModel,
    bias: bool = False,
) -> Model:
    """Construct a public-facing model from an internal model representation.

    Args:
        internal_model: the internal model.
        bias: whether or not the model contains a bias.

    Returns:
        A public-facing model with identical state.
    """
    model: Model
    if isinstance(internal_model, GatedReLUMLP):
        G, G_bias = extract_gates_bias(internal_model.U, bias)
        model = NonConvexGatedReLU(
            G, internal_model.c, bias=bias, G_bias=G_bias
        )
        w1, w2 = internal_model._split_weights(internal_model.weights)
        parameters = extract_bias(w1, bias) + extract_bias(w2, False)
        model.set_parameters(parameters)

    elif isinstance(internal_model, ReLUMLP):
        d = internal_model.d
        if bias:
            d = d - 1
        model = NonConvexReLU(d, internal_model.p, internal_model.c, bias=bias)
        w1, w2 = internal_model._split_weights(internal_model.weights)
        parameters = extract_bias(w1, bias) + extract_bias(w2, False)

        model.set_parameters(parameters)
    elif isinstance(internal_model, LinearRegression):
        model = LinearModel(internal_model.d, internal_model.c, bias=bias)
        model.parameters = extract_bias(internal_model.weights, bias)
    else:
        raise ValueError(f"Model {internal_model} not supported.")

    return model
