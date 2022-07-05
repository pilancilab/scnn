"""Generate activation patterns for convex reformulations of neural networks.

Overview:
    This module provides functions for generating gate vectors and activation
    patterns for ReLU or threshold activation functions. These patterns are
    used when forming (subsampled) convex reformulations of neural networks
    with those activations. Gate vectors are also required for Gated ReLU
    networks. 

    An activation pattern is a vector of the form,

    .. math:: d_i = 1(X w \\geq 0),

    where :math:`1(z \\geq 0)` is an element-wise indicator function whose
    i'th element is one when :math:`z_i \\geq 0` and zero-otherwise and
    :math:`w \\in \\mathbb{R}^d` is a "gate vector".
    Forming the convex reformulation of a neural network with ReLU activations
    requires enumerating the activation patterns a single ReLU or threshold
    neuron can take on,

    .. math:: \\mathcal{D} = \\left\\{  d = 1(X w \\geq 0) : w \\in
        \\mathbb{R}^d \\right\\}.

    In practice, :math`\\mathcal{D}` can approximated sampling vectors
    :math:`w \\sim P` according to some distribution :math:`P` and then 
    computing the corresponding pattern :math:`d_i`.

TODO:
    - Where to put ugly conv mask code?
"""

from typing import Tuple, List, Optional
from typing_extensions import Literal
import math
from itertools import combinations

import numpy as np
from numpy.random import Generator


GateType = Literal[
    "dense",
    "feature_sparse",
    "convolutional",
]


def sample_gate_vectors(
    seed: int,
    d: int,
    n_samples: int,
    gate_type: GateType = "dense",
    order: Optional[int] = None,
) -> np.ndarray:
    """Generate gate vectors by random sampling.

    Args:
        seed: the random seed to use when generating the gates.
        d: the dimensionality of the gate vectors.
        n_samples: the number of samples to use.
        gate_type: the type of gates to sample. Must be one of:

            - `'dense'`: sample dense gate vectors.

            - `'feature_sparse'`: sample gate vectors which are sparse in
                specific features.

        order: the maximum order of feature sparsity to consider.
            Only used for `gate_type='feature_sparse'`. See
            :func:`sample_sparse_gates` for more details.

    Notes:
        It is possible to obtain more than `n_samples` gate vectors when
            `gate_type='feature_sparse'`
        if `len(sparsity_indices)` does not divide evenly into n_samples. In
            this case, we use the ceiling function to avoid sampling zero
            gates for some sparsity patterns.
    """

    rng = np.random.default_rng(seed)

    if gate_type == "dense":
        G = sample_dense_gates(rng, d, n_samples)
    elif gate_type == "feature_sparse":
        if order is None:
            raise ValueError(
                "`Order` must be specified when sampling feature-sparse gate."
            )
        sparsity_indices = generate_index_lists(d, order)
        G = sample_sparse_gates(rng, d, n_samples, sparsity_indices)
    elif gate_type == "convolutional":
        G = sample_convolutional_gates(rng, d, n_samples)
    else:
        raise ValueError(f"Gate type {gate_type} not recognized.")

    return G


def sample_dense_gates(
    rng: np.random.Generator,
    d: int,
    n_samples: int,
) -> np.ndarray:
    """Generate dense gate vectors by random sampling.

    Args:
        rng: a NumPy random number generator.
        d: the dimensionality of the gate vectors.
        n_samples: the number of samples to use.

    Returns:
        G -  a :math:`d \\times \\text{n_samples}` matrix of gate vectors.
    """
    G = rng.standard_normal((d, n_samples))

    return G


def sample_sparse_gates(
    rng: np.random.Generator,
    d: int,
    n_samples: int,
    sparsity_indices: List[List[int]],
) -> np.ndarray:
    """Generate feature-sparse gate vectors by random sampling.

    Args:
        rng: a NumPy random number generator.
        d: the dimensionality of the gate vectors.
        n_samples: the number of samples to use.
        sparsity_indices: lists of indices (i.e. features) for which sparse
            gates should be generated. Each index list will get
            `n_samples / len(sparsity_indices)` gates which are sparse in
            every feature except the given indices.

    Notes:
        - It is possible to obtain more than `n_samples` gate vectors if
            `len(sparsity_indices)` does not divide evenly into n_samples.
            In this case, we use the ceiling to avoid sampling zero gates for
            some sparsity patterns.

    Returns:
        G -  a :math:`d \\times \\text{n_samples}` matrix of gate vectors.
    """
    samples_per_list = math.ceil(n_samples / len(sparsity_indices))
    gates = []

    for indices in sparsity_indices:
        # create mask
        mask = np.zeros((d, 1))
        mask[indices, :] = 1

        G = sample_dense_gates(rng, d, samples_per_list)

        # project gates onto subspace
        gates.append(np.multiply(G, mask))

    return np.concatenate(gates, axis=1)


def sample_convolutional_gates(
    rng: np.random.Generator,
    d: int,
    n_samples: int,
) -> np.ndarray:
    """Generate convolutional gate vectors by random sampling.

    Args:
        rng: a NumPy random number generator.
        d: the dimensionality of the gate vectors.
        n_samples: the number of samples to use.

    Returns:
        G -  a :math:`d \\times \\text{n_samples}` matrix of gate vectors.
    """
    G = sample_dense_gates(rng, d, n_samples)

    if d == 784:
        conv_masks = _generate_conv_masks(rng, n_samples, 28, 1)
    elif d == 3072:
        conv_masks = _generate_conv_masks(rng, n_samples, 32, 3)
    else:
        assert (
            False
        ), "Convolutional patterns only implemented for MNIST or CIFAR datasets"

    G = G * conv_masks
    return G


def compute_activation_patterns(
    X: np.ndarray,
    G: np.ndarray,
    filter_duplicates: bool = True,
    filter_zero: bool = True,
    bias: bool = False,
    active_proportion: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation patterns corresponding to a set of gate vectors.

    Args:
        X: an (n x d) data matrix, where the examples are stored as rows.
        G: an (d x m) matrix of "gate vectors" used to generate the activation
            patterns.
        filter_duplicates: whether or not to remove duplicate activation
            patterns and the corresponding
        filter_zero: whether or not to filter the zero activation pattern and
            corresponding gates. Defaults to `True`.
        bias: whether or not a bias should be included in the gate vectors.
        active_proportion: force each gate to be active for
            `active_proportion`*n of the training examples using a bias
            term. This feature is only supported when `bias == True`.

    Returns:
        - `D`, an (n x p) matrix of (possibly unique) activation patterns where
            :math:`p \\leq m`

        - `G`, a (d x b) matrix of gate vectors generating `D`.
    """
    n, d = X.shape

    # need to extend the gates with zeros.
    if bias and G.shape[0] + 1 == X.shape[1]:
        G = np.concatenate([G, np.zeros((1, G.shape[1]))], axis=0)

    XG = np.matmul(X, G)

    if active_proportion is not None:
        # Gates must be augmented with a row of zeros to be valid.
        assert np.all(G[-1] == 0)
        # X must be augmented with a column of ones to be valid.
        assert np.all(X[:, -1] == X[0, -1])

        # set bias terms in G
        quantiles = np.quantile(XG, q=1 - active_proportion, axis=0, keepdims=True)
        XG = XG - quantiles
        G = G.copy()
        G[-1] = -np.ravel(quantiles)

    XG = np.maximum(XG, 0)
    XG[XG > 0] = 1

    if filter_duplicates:
        D, indices = np.unique(XG, axis=1, return_index=True)
        G = G[:, indices]

    # filter out the zero column.
    if filter_zero:
        non_zero_cols = np.logical_not(np.all(D == np.zeros((n, 1)), axis=0))
        D = D[:, non_zero_cols]
        G = G[:, non_zero_cols]

    return D, G


def generate_index_lists(
    d: int,
    order: int,
) -> List[List[int]]:
    """Generate lists of all groups of indices of order up to and including
    `order`.

    Args:
        d: the dimensionality or maximum index.
        order: the order to which the lists should be generated.
            For example, `order=2` will generate all singleton lists and
            all lists of pairs of indices.

    Returns:
        List of list of indices.
    """
    assert order > 0

    index_lists: List[List[int]] = []
    all_indices = list(range(d))

    for i in range(1, order + 1):
        d_choose_i = [list(comb) for comb in combinations(all_indices, i)]
        index_lists = index_lists + d_choose_i

    return index_lists


def _generate_conv_masks(
    rng: Generator,
    num_samples: int,
    image_size: int = 32,
    channels: int = 3,
    kernel_size: int = 3,
):
    upper_left_coords = rng.integers(
        low=0, high=image_size - kernel_size - 1, size=(num_samples, 2)
    )
    upper_left_indices = image_size * upper_left_coords[:, 0] + upper_left_coords[:, 1]
    upper_rows = [
        np.arange(upper_left_indices[i], upper_left_indices[i] + kernel_size)
        for i in range(num_samples)
    ]
    first_patch = [
        np.concatenate(
            [
                np.arange(
                    upper_rows[i][j],
                    upper_rows[i][j] + kernel_size * image_size,
                    image_size,
                )
                for j in range(kernel_size)
            ]
        )
        for i in range(num_samples)
    ]
    all_patches = [
        np.concatenate(
            [
                np.arange(
                    first_patch[i][j],
                    first_patch[i][j] + channels * image_size**2,
                    image_size**2,
                )
                for j in range(kernel_size**2)
            ]
        ).tolist()
        for i in range(num_samples)
    ]
    mask = np.zeros((num_samples, channels * image_size**2))

    def _transpose(x):
        dims = np.arange(len(x.shape))
        dims[[0, 1]] = [1, 0]

        return np.transpose(x, dims)

    mask[np.arange(num_samples), _transpose(all_patches)] = 1.0

    return mask.t()
