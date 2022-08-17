"""Solution mappings for the C-ReLU and C-GReLU problems."""

import lab


def grelu_solution_mapping(weights, G, remove_sparse: bool = False):
    assert len(weights.shape) == 3

    weight_norms = (lab.sum(weights**2, axis=-1, keepdims=True)) ** (1 / 4)
    normalized_weights = lab.safe_divide(weights, weight_norms)

    first_layer = None
    second_layer = []
    for c in range(weights.shape[0]):
        pre_zeros = [
            lab.zeros_like(weight_norms[0]) for i in range(c)
        ]  # positive neurons
        post_zeros = [
            lab.zeros_like(weight_norms[0]) for i in range(weights.shape[0] - c - 1)
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
    U = lab.concatenate([G for c in range(weights.shape[0])], axis=1)

    if remove_sparse:
        sparse_indices = lab.sum(first_layer, axis=-1) != 0

        first_layer = first_layer[sparse_indices]
        second_layer = second_layer[:, sparse_indices]

        U = U[:, sparse_indices]

    return (first_layer, second_layer, U)


def relu_solution_mapping(weights, G, remove_sparse: bool = False):
    assert len(weights.shape) == 4

    weight_norms = (lab.sum(weights**2, axis=-1, keepdims=True)) ** (1 / 4)
    normalized_weights = lab.safe_divide(weights, weight_norms)

    num_classes = weights.shape[1]
    first_layer = None
    second_layer = []
    for c in range(num_classes):
        pre_zeros = [
            lab.zeros_like(weight_norms[0, c]) for i in range(2 * c)
        ]  # positive neurons
        post_zeros = [
            lab.zeros_like(weight_norms[0, c]) for i in range(2 * (num_classes - c - 1))
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
