"""Construct efficient operators for convex neural networks."""
from typing import Tuple

from .kernels import einsum_kernel as ek
from .kernels import direct_kernel as dk

# constants

EINSUM = "einsum"
DIRECT = "direct"

KERNELS = [EINSUM, DIRECT]

# functions

expanded_data_matrix = dk.expanded_data_matrix
expanded_hessian = dk.expanded_hessian
expanded_bd_hessian = dk.expanded_bd_hessian


def get_kernel(kernel_name: str) -> Tuple:
    """Get a complete set of "kernel" functions for models.nns.ConvexMLP. These
    kernel functions implement operations with the expanded data matrix.

    .. math:: A = [D_1 X, D_2 X, ..., D_P X],
    without (necessarily) forming A directly. Note that these *only* work with the squared
    loss at the moment.
    :params kernel_name: the kernel to use. Current implemented kernels are
    EINSUM: compute operations without forming A using `einsum`; and
    DIRECT: form A and use the matrix directly.
    EINSUM is *much* more efficient than DIRECT and should be preferred except
    when running comparisons.
    :returns: tuple of Callable (data_mvp, data_t_mvp, grad, hessian_mvp, bd_hessian_mvp), where
    'data_mvp' computes products of the form :math:`A v`; and
    'data_t_mvp' computes transpose products A:math:`A.T w`; and
    'hessian_mvp' computes Hessian-vector products :math:`H v`; and
    'bd_hessian_mvp' computes Hessian-vector products with a block-diagonal
    approximation of the Hessian.
    """

    if kernel_name == EINSUM:
        kernel_funcs = einsum_kernel()
    elif kernel_name == DIRECT:
        kernel_funcs = direct_kernel()

    return kernel_funcs


def get_matrix_builders(kernel_name: str) -> Tuple:
    """Get a complete set of "matrix builders" for models.nns.ConvexMLP. These
    functions directly construct the matrices associated with the model, such
    as the expanded data matrix,

    .. math:: A = [D_1 X, D_2 X, ..., D_P X],
    without (necessarily) computing intermediate products. Note that these *only* work
    with the squared loss at the moment.
    :params kernel_name: the kernel to use. Current implemented kernels are
    EINSUM: compute matrices using `einsum`; and
    DIRECT: form the matrices by (slow) direct computation.
    EINSUM is *much* more efficient than DIRECT, which serves only as a reference
    implementation.
    :returns: tuple of Callable (data, hessian, bd_hessian), where
    'data_builder' computes the expanded data matrix :math:`A`;
    'hessian' computes the Hessian of the squared loss, :math:`A.T A`; and
    'bd_hessian' computes the diagonal blocks of the Hessian.
    """

    if kernel_name == EINSUM:
        builders = einsum_matrix_builders()
    elif kernel_name == DIRECT:
        builders = direct_matrix_builders()

    return builders


def einsum_kernel() -> Tuple:
    """Get "kernel" operators for a ConvexMLP with squared loss based on
    einsum.

    These operators are quite cryptic, but often an order of magnitude faster than
    direct computation.
    :returns: tuple of Callable (data_mvp, data_t_mvp, grad, hessian_mvp, bd_hessian_mvp), where
    'data_mvp' computes products of the form :math:`A v`;
    'data_t_mvp' computes transpose products :math:`A.T w`;
    'hessian_mvp' computes Hessian-vector products :math:`H v`; and
    'bd_hessian_mvp' computes Hessian-vector products with a block-diagonal
    approximation of the Hessian.
    """

    return (
        ek.data_mvp,
        ek.data_t_mvp,
        ek.gradient,
        ek.hessian_mvp,
        ek.bd_hessian_mvp,
    )


def einsum_matrix_builders() -> Tuple:
    """Get matrix builders for a ConvexMLP with squared loss based on einsum.

    These functions are quite cryptic, but often an order of magnitude faster than
    direct computation.
    :returns: tuple of Callable (data, hessian, bd_hessian), where
    'data_builder' computes the expanded data matrix :math:`A`;
    'hessian' computes the Hessian of the squared loss, :math:`A.T A`; and
    'bd_hessian' computes the diagonal blocks of the Hessian.
    """

    return ek.data, ek.hessian, ek.bd_hessian


# Direct implementations


def direct_matrix_builders():
    """Get matrix builders for a ConvexMLP with squared loss based on einsum.

    These functions are compute the matrices slowly and directly and are only
    meant as reference implementations.
    :returns: tuple of Callable (data, hessian, bd_hessian), where
    'data_builder' computes the expanded data matrix :math:`A`;
    'hessian' computes the Hessian of the squared loss, :math:`A.T A`; and
    'bd_hessian' computes the diagonal blocks of the Hessian.
    """

    return dk.data, dk.hessian, dk.bd_hessian


def direct_kernel():
    """Get "direct" operators for a ConvexMLP with squared loss based on
    einsum.

    These operators compute quantities by directly forming the expanded data matrix,
    which is slow but reliable. They are intended only for comparisons or as a reference.
    :returns: tuple of Callable (data_mvp, data_t_mvp, grad, hessian_mvp, bd_hessian_mvp), where
    'data_mvp' computes products of the form :math:`A v`;
    'data_t_mvp' computes transpose products :math:`A.T w`;
    'hessian_mvp' computes Hessian-vector products :math:`H v`; and
    'bd_hessian_mvp' computes Hessian-vector products with a block-diagonal
    approximation of the Hessian.
    """

    # pre-compute expanded matrix and hessian.

    return dk.data_mvp, dk.data_t_mvp, dk.gradient, dk.hessian_mvp, dk.bd_hessian_mvp
