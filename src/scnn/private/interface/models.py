"""Convert models from :module:`scnn.models` into internal representations
and vice versa."""

from typing import Optional, List, Tuple

import numpy as np

import lab

from scnn.regularizers import (
    Regularizer,
    NeuronGL1,
    SkipNeuronGL1,
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
    SkipMLP,
    AL_MLP,
    SkipALMLP,
    GroupL1Regularizer,
    SkipGroupL1Regularizer,
    FeatureGroupL1Regularizer,
    L2Regularizer,
    L1Regularizer,
    LinearRegression,
    grelu_solution_mapping,
    relu_solution_mapping,
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
    elif isinstance(regularizer, SkipNeuronGL1):
        reg = SkipGroupL1Regularizer(lam, regularizer.skip_lam)
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

    G_input = model.G

    if model.bias:
        G_input = np.concatenate([G_input,model.G_bias.reshape([1,-1])],axis=0)

    D, G = lab.all_to_tensor(
        compute_activation_patterns(
            lab.to_np(X_train),
            G_input,
            bias=model.bias,
        ),
        dtype=lab.get_dtype(),
    )

    if isinstance(model, ConvexReLU):
        if model.skip_connection:
            model_class = SkipALMLP
        else:
            model_class = AL_MLP

        internal_model = model_class(
            d,
            D,
            G,
            "einsum",
            delta=1000,
            regularizer=internal_reg,
            c=c,
        )
    elif isinstance(model, ConvexGatedReLU):
        if model.skip_connection:
            model_class = SkipMLP
        else:
            model_class = ConvexMLP

        internal_model = model_class(d, D, G, "einsum", regularizer=internal_reg, c=c)
    else:
        raise ValueError(f"Model object {model} not supported.")

    internal_model._bias = model.bias
    internal_model._skip_connection = model.skip_connection

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
    p = G.shape[-1]

    if bias:
        return (G[0:-1], G[-1])
    else:
        return (G, np.zeros(p))


def extract_skip_connection(
    internal_model: InternalModel, skip_connection: bool = False
) -> Tuple[lab.Tensor, List[np.ndarray]]:

    weights = internal_model.weights
    skip_weights = []
    if skip_connection:
        weights, sw = internal_model.get_weights()
        skip_weights = extract_bias(sw, bias=internal_model._bias)

    return weights, skip_weights


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

        weights, skip_weights = extract_skip_connection(
            internal_model,
            model.skip_connection,
        )

        model.set_parameters(
            extract_bias(weights, model.bias) + skip_weights,
        )
        model.G, model.G_bias = extract_gates_bias(
            internal_model.U,
            model.bias,
        )
    elif isinstance(model, ConvexReLU):
        assert isinstance(internal_model, AL_MLP)

        weights, skip_weights = extract_skip_connection(
            internal_model,
            model.skip_connection,
        )

        model.set_parameters(
            extract_bias(weights[0], model.bias)
            + extract_bias(weights[1], model.bias)
            + skip_weights
        )

        model.G, model.G_bias = extract_gates_bias(internal_model.U, model.bias)
    elif isinstance(model, LinearModel):
        model.set_parameters(extract_bias(internal_model.weights, model.bias))

    return model


def get_nc_formulation(
    internal_model: InternalModel,
) -> Model:
    """Construct a public-facing model from an internal model representation.

    Args:
        internal_model: the internal model.
        bias: whether or not the model contains a bias.

    Returns:
        A public-facing model with identical state.
    """
    bias = internal_model._bias
    skip_connection = internal_model._skip_connection

    nc_model: Model

    if not isinstance(internal_model, AL_MLP):

        weights, skip_weights = extract_skip_connection(
            internal_model,
            skip_connection,
        )

        w1, w2, G = grelu_solution_mapping(
            weights,
            internal_model.U,
            remove_sparse=True,
        )

        G, G_bias = extract_gates_bias(G, bias)
        nc_model = NonConvexGatedReLU(
            G,
            internal_model.c,
            bias=bias,
            G_bias=G_bias,
            skip_connection=skip_connection,
        )

        parameters = extract_bias(w1, bias) + extract_bias(w2, False)
        nc_model.set_parameters(parameters + skip_weights)

    elif isinstance(internal_model, AL_MLP):
        d = internal_model.d

        if bias:
            d = d - 1

        weights, skip_weights = extract_skip_connection(
            internal_model,
            skip_connection,
        )

        w1, w2 = relu_solution_mapping(
            weights,
            internal_model.U,
            remove_sparse=True,
        )
        nc_model = NonConvexReLU(
            d, w1.shape[0], internal_model.c, bias=bias, skip_connection=skip_connection
        )
        parameters = extract_bias(w1, bias) + extract_bias(w2, False)
        nc_model.set_parameters(parameters + skip_weights)

    elif isinstance(internal_model, LinearRegression):
        nc_model = LinearModel(internal_model.d, internal_model.c, bias=bias)
        nc_model.parameters = extract_bias(internal_model.weights, bias)
    else:
        raise ValueError(f"Model {internal_model} not supported.")

    return nc_model
