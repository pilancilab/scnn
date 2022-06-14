""""Direct" operators for convex neural networks based on manually constructing
the underlying matrices."""
from typing import Optional

import lab

# operators


def data_mvp(
    v: lab.Tensor,
    X: lab.Tensor,
    D: lab.Tensor,
    expanded_X: Optional[lab.Tensor] = None,
    optimize: Optional[bool] = None,
) -> lab.Tensor:
    if expanded_X is None:
        expanded_X = expanded_data_matrix(X, D)
    w = v.reshape(-1, X.shape[1]*D.shape[1])
    return expanded_X @ w.T


def data_t_mvp(
    v: lab.Tensor,
    X: lab.Tensor,
    D: lab.Tensor,
    expanded_X: Optional[lab.Tensor] = None,
    optimize: Optional[bool] = None,
) -> lab.Tensor:
    if expanded_X is None:
        expanded_X = expanded_data_matrix(X, D)
    w = v.reshape(-1)
    return expanded_X.T @ w


def gradient(
    v: lab.Tensor,
    X: lab.Tensor,
    y: lab.Tensor,
    D: lab.Tensor,
    expanded_X: Optional[lab.Tensor] = None,
    optimize: Optional[bool] = None,
) -> lab.Tensor:
    if expanded_X is None:
        expanded_X = expanded_data_matrix(X, D)

    return (expanded_X.T @ (data_mvp(v, X, D, expanded_X) - y)).T.reshape(*v.shape)


def hessian_mvp(
    v: lab.Tensor,
    X: lab.Tensor,
    D: lab.Tensor,
    hessian: Optional[lab.Tensor] = None,
    optimize: Optional[bool] = None,
) -> lab.Tensor:

    if hessian is None:
        hessian = expanded_hessian(X, D)
        # flatten the Hessian.

        hessian = [entry for entry in hessian]
        hessian = lab.concatenate(hessian, axis=1)
        hessian = [entry for entry in hessian]
        hessian = lab.concatenate(hessian, axis=1)

    w = v.reshape(-1, X.shape[1]*D.shape[1])
    return (hessian @ w.T).T.reshape(*v.shape)


def bd_hessian_mvp(
    v: lab.Tensor,
    X: lab.Tensor,
    D: lab.Tensor,
    bd_hessian: Optional[lab.Tensor] = None,
    optimize: Optional[bool] = None,
) -> lab.Tensor:
    if bd_hessian is None:
        bd_hessian = expanded_bd_hessian(X, D)

    w = v.reshape(-1, D.shape[1], X.shape[1])
    res = []

    for i, block in enumerate(bd_hessian):
        res.append((block @ w[:, i].T).T)

    return lab.transpose(lab.stack(res), 0, 1).reshape(*v.shape)


# builders


def data(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return expanded_data_matrix(X, D)


def hessian(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return expanded_hessian(X, D)


def bd_hessian(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return expanded_bd_hessian(X, D)


# helpers for direct matrix operations.


def expanded_data_matrix(
    X: lab.Tensor,
    D: lab.Tensor,
) -> lab.Tensor:
    """Construct the expanded data matrix A = [D_1 X, D_2 X, ..., D_p X] associated
    with the linear model,
        $sum_i D_i X v_i$
    where v_i is the i'th block of the input vector 'v'. Note: direct computation of A
    is not efficient; whenever possible, use the einsum kernels to construct an operator
    which can evaluate $A v$ and $A.T w$.
    :param X: (n,d) array containing the data examples.
    :param D: array of possible sign patterns.
    :returns: expanded data matrix .
    """
    P = D.shape[1]
    blocks = []
    for i in range(P):
        res = lab.diag(D[:, i]) @ X
        blocks.append(res)

    return lab.concatenate(blocks, axis=1)


def expanded_hessian(X: lab.Tensor, D: lab.Tensor, flatten: bool = False) -> lab.Tensor:
    """Construct Hessian of the squared loss for the linear model.

        $sum_i D_i X v_i$
    where v_i is the i'th block of the input vector 'v'. Note: direct computation
    is not efficient; whenever possible, use the einsum kernels to construct an operator
    which can evaluate $H v$ instead of calling this function.
    :param X: (n,d) array containing the data examples.
    :param D: array of possible sign patterns.
    :param flatten: whether or not to flatten the output matrix.
    :returns: Hessian of the expanded linear model.
    """

    P = D.shape[1]

    # compute matrix blocks by hand
    rows = []
    for i in range(P):
        blocks = []
        for j in range(P):
            H_ij = X.T @ lab.diag(D[:, i]) @ lab.diag(D[:, j]) @ X

            blocks.append(H_ij)

        rows.append(lab.stack(blocks))

    return lab.stack(rows)


def expanded_bd_hessian(
    X: lab.Tensor,
    D: lab.Tensor,
) -> lab.Tensor:
    """Construct the diagonal blocks H_i of the Hessian of the squared loss for
    the linear model.

        $sum_i D_i X v_i$
    where v_i is the i'th block of the input vector 'v'. Note: direct computation
    is not efficient; whenever possible, use the einsum kernels to construct an operator
    which can evaluate $H v$ instead of calling this function.
    :param X: (n,d) array containing the data examples.
    :param D: array of possible sign patterns.
    :returns: array with shape (P, d, d), where the first axis indexes the blocks.
    """
    P = D.shape[1]

    # compute matrix blocks by hand
    diagonal = []
    for i in range(P):
        H_ii = X.T @ lab.diag(D[:, i]) @ lab.diag(D[:, i]) @ X

        diagonal.append(H_ii)

    return lab.stack(diagonal)
