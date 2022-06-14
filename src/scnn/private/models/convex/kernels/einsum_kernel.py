"""Einsum-based operators for convex neural networks."""

import numpy as np

import lab


def data_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return lab.einsum("ij, lkj, ik->il", X, w, D)


def data_t_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return lab.einsum("ij, i, il->jl", D, v, X).reshape(-1)


# TODO: accept residual as a parameter and remove data_mvp from this operation.
def gradient(
    v: lab.Tensor,
    X: lab.Tensor,
    y: lab.Tensor,
    D: lab.Tensor,
) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return lab.einsum("ij, il, ik->ljk", D, data_mvp(w, X, D) - y, X).reshape(*v.shape)


def hessian_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return lab.einsum("ij, nkj, ik, il, im ->nlm", X, w, D, D, X).reshape(*v.shape)


def bd_hessian_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return lab.einsum("ij, nkj, ik, il->nkl", X, w, D, X).reshape(*v.shape)


# builders


def data(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return np.concatenate(lab.einsum("ij, ik->jik", D, X), axis=1)


def hessian(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return lab.einsum("ij, ik, il, im -> jkml", D, D, X, X)


def bd_hessian(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return lab.einsum("ij, il, im -> jml", D, X, X)
