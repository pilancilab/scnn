"""Preconditioners for iterative linear-system solvers."""

from typing import Optional, Union

from scnn.private.utils.linear_operators import MatVecOperator
import lab


def get_preconditioner(
    name: str, X: lab.Tensor, D: Optional[lab.Tensor] = None
) -> Union[MatVecOperator, lab.Tensor]:
    """Lookup and return preconditioner.

    :param name: the name of the preconditioner to return.
    :param X: (n,d) feature matrix of training examples.
    :param D: (optional) a matrix of sign patterns for the expanded
    feature matrix.
    """
    if name == "column_norm":
        return column_norm(X, D)
    elif name == "H_diag":
        n = X.shape[0]
        return hessian_diagonal(X, D) / n
    else:
        raise ValueError(f"Preconditioner {name} not recognized!")


def hessian_diagonal(X: lab.Tensor, D: Optional[lab.Tensor] = None) -> lab.Tensor:
    """Compute the diagonal of the Hessian for the squared loss. If D is not
    None, then the diagonal the expanded matrix.

    .. math::

        [diag(D[:,1] X, ...., diag(D[:,P]) X],

    is computed.
    :param X: the (n,d) matrix of data examples.
    :param D: (optional) a (d, p) associated with a convex neural network. See 'models/nns.py'.
    :returns: diagonal of the Hessian matrix as a vector.
    """

    if D is not None:
        # compute column norms of expanded matrix.
        H_diag = lab.einsum("ij, ik->jk", D, X ** 2)
    else:
        H_diag = lab.sum(X ** 2, axis=0)

    return H_diag


def column_norm(X: lab.Tensor, D: Optional[lab.Tensor] = None) -> MatVecOperator:
    """Compute a diagonal preconditioner that unitizes the columns of 'X', as described
    here: https://web.stanford.edu/group/SOL/software/lsmr/.
    If D is not None, then a preconditioner for the expanded matrix
    .. math::

        [diag(D[:,1] X, ...., diag(D[:,P]) X],

    is computed.
    :param X: the (n,d) matrix of data examples.
    :param D: (optional) a (d, p) associated with a convex neural network. See 'models/nns.py'.
    :returns: (forward_map, inverse_map, column_norms)
    """

    H_diag = hessian_diagonal(X, D).reshape(-1)
    col_norms = lab.sqrt(H_diag)

    # create operators for forward and inverse maps.
    def forward_map(v: lab.Tensor):
        v = v.reshape(-1)
        return lab.safe_divide(v, col_norms)

    s = col_norms.shape[0]

    # both operators are symmetric and real, so the Hermitian/transpose is the same operator.
    forward = MatVecOperator(shape=(s, s), forward=forward_map, transpose=forward_map)

    return forward
