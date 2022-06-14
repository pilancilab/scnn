"""Solution mappings for the C-ReLU and C-GReLU problems."""
import torch

import lab

from scnn.private.models.convex import ConvexMLP
from scnn.private.models.non_convex import (
    GatedReLULayer,
    GatedReLUMLP,
    ReLUMLP,
)
from scnn.private.models.regularizers.l2 import L2Regularizer


def is_grelu_compatible(torch_model: torch.nn.Module) -> bool:
    submodules = list(torch_model.children())

    return (
        len(submodules) == 2
        and isinstance(submodules[0], GatedReLULayer)
        and submodules[0].linear.bias is None
        and isinstance(submodules[1], torch.nn.Linear)
        and submodules[1].bias is None
    )


def is_relu_compatible(torch_model: torch.nn.Module) -> bool:
    submodules = list(torch_model.children())

    return (
        len(submodules) == 3
        and isinstance(submodules[0], torch.nn.Linear)
        and submodules[0].bias is None
        and isinstance(submodules[1], torch.nn.ReLU)
        and isinstance(submodules[2], torch.nn.Linear)
        and submodules[2].bias is None
    )


def grelu_solution_mapping(convex_model, remove_sparse: bool = False):
    weights = convex_model.weights
    assert len(weights.shape) == 3

    weight_norms = (lab.sum(weights ** 2, axis=-1, keepdims=True)) ** (1 / 4)
    normalized_weights = lab.safe_divide(weights, weight_norms)

    first_layer = None
    second_layer = []
    for c in range(weights.shape[0]):
        pre_zeros = [
            lab.zeros_like(weight_norms[0]) for i in range(c)
        ]  # positive neurons
        post_zeros = [
            lab.zeros_like(weight_norms[0])
            for i in range(weights.shape[0] - c - 1)
        ]

        if first_layer is None:
            pre_weights = []
        else:
            pre_weights = [first_layer]

        first_layer = lab.concatenate(
            pre_weights
            + [
                normalized_weights[c],
            ],
            axis=0,
        )

        w2 = lab.concatenate(
            pre_zeros
            + [
                weight_norms[c],
            ]
            + post_zeros,
            axis=0,
        ).T
        second_layer.append(w2)
    second_layer = lab.concatenate(second_layer, axis=0)
    U = lab.concatenate(
        [convex_model.U for c in range(weights.shape[0])], axis=1
    )

    if remove_sparse:
        sparse_indices = lab.sum(first_layer, axis=-1) != 0

        first_layer = first_layer[sparse_indices]
        second_layer = second_layer[:, sparse_indices]

        U = U[:, sparse_indices]

    return (first_layer, second_layer, U)


def relu_solution_mapping(convex_model, remove_sparse: bool = False):
    weights = convex_model.weights
    assert len(weights.shape) == 4

    weight_norms = (lab.sum(weights ** 2, axis=-1, keepdims=True)) ** (1 / 4)
    normalized_weights = lab.safe_divide(weights, weight_norms)

    num_classes = weights.shape[1]
    first_layer = None
    second_layer = []
    for c in range(num_classes):
        pre_zeros = [
            lab.zeros_like(weight_norms[0, c]) for i in range(2 * c)
        ]  # positive neurons
        post_zeros = [
            lab.zeros_like(weight_norms[0, c])
            for i in range(2 * (num_classes - c - 1))
        ]

        if first_layer is None:
            pre_weights = []
        else:
            pre_weights = [first_layer]

        first_layer = lab.concatenate(
            pre_weights
            + [
                normalized_weights[0][c],
                normalized_weights[1][c],
            ],
            axis=0,
        )

        w2 = lab.concatenate(
            pre_zeros
            + [
                weight_norms[0][c],
                -weight_norms[1][c],
            ]
            + post_zeros,
            axis=0,
        ).T
        second_layer.append(w2)

    second_layer = lab.concatenate(second_layer, axis=0)

    if remove_sparse:
        sparse_indices = lab.sum(first_layer, axis=1) != 0

        first_layer = first_layer[sparse_indices]
        second_layer = second_layer[:, sparse_indices]

    return first_layer, second_layer


def convex_mlp_to_torch_mlp(
    convex_model: ConvexMLP,
    torch_model: torch.nn.Module,
    grelu: bool = False,
    remove_sparse: bool = False,
):
    submodules = list(torch_model.children())
    if grelu:
        assert is_grelu_compatible(torch_model)

        first_layer, second_layer, _ = grelu_solution_mapping(
            convex_model, remove_sparse
        )
        (submodules[0].linear.weight.data, submodules[1].weight.data) = (
            lab.torch_backend.torch_tensor(first_layer),
            lab.torch_backend.torch_tensor(second_layer),
        )
    else:
        assert is_relu_compatible(torch_model)

        first_layer, second_layer = relu_solution_mapping(
            convex_model, remove_sparse
        )

        (submodules[0].weight.data, submodules[2].weight.data,) = (
            lab.torch_backend.torch_tensor(first_layer),
            lab.torch_backend.torch_tensor(second_layer),
        )

    return torch_model


def construct_nc_torch(
    convex_model: ConvexMLP,
    grelu: bool = False,
    remove_sparse: bool = False,
):

    if grelu:
        first_layer, second_layer, U = grelu_solution_mapping(
            convex_model, remove_sparse
        )
        first_layer, second_layer, U = (
            lab.torch_backend.torch_tensor(first_layer),
            lab.torch_backend.torch_tensor(second_layer),
            lab.torch_backend.torch_tensor(U),
        )

        torch_model = torch.nn.Sequential(
            GatedReLULayer(U),
            torch.nn.Linear(U.shape[1], 1, bias=False),
        )
        submodules = list(torch_model.children())
        (submodules[0].linear.weight.data, submodules[1].weight.data,) = (
            first_layer,
            second_layer,
        )
    else:
        first_layer, second_layer = relu_solution_mapping(
            convex_model, remove_sparse
        )
        first_layer, second_layer = (
            lab.torch_backend.torch_tensor(first_layer),
            lab.torch_backend.torch_tensor(second_layer),
        )
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(
                first_layer.shape[1], first_layer.shape[0], bias=False
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                second_layer.shape[1], second_layer.shape[0], bias=False
            ),
        )
        submodules = list(torch_model.children())
        (submodules[0].weight.data, submodules[2].weight.data,) = (
            first_layer,
            second_layer,
        )

    return torch_model


def convex_mlp_to_manual_mlp(
    convex_model: ConvexMLP,
    manual_model: ReLUMLP,
    grelu: bool = False,
    remove_sparse: bool = False,
):
    if grelu:
        first_layer, second_layer, U = grelu_solution_mapping(
            convex_model, remove_sparse
        )
        assert isinstance(manual_model, GatedReLUMLP)

        manual_model.weights = manual_model._join_weights(
            first_layer, second_layer
        )
        manual_model.U = U
        manual_model.p = U.shape[1]
    else:
        first_layer, second_layer = relu_solution_mapping(
            convex_model, remove_sparse
        )
        manual_model.p = first_layer.shape[0]
        manual_model.weights = manual_model._join_weights(
            first_layer, second_layer
        )

    return manual_model


def construct_nc_manual(
    convex_model: ConvexMLP,
    grelu: bool = False,
    remove_sparse: bool = False,
):

    if grelu:
        manual_model = GatedReLUMLP(
            convex_model.d,
            convex_model.U,
            c=convex_model.c,
        )
    else:
        manual_model = ReLUMLP(
            convex_model.d, convex_model.p, c=convex_model.c
        )

    nc_model = convex_mlp_to_manual_mlp(
        convex_model, manual_model, grelu, remove_sparse
    )

    l2_regularizer = None
    if convex_model.regularizer is not None:
        l2_regularizer = L2Regularizer(convex_model.regularizer.lam)

    nc_model.regularizer = l2_regularizer

    return nc_model
