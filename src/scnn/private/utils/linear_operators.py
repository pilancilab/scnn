"""Block-diagonal matrix."""
from typing import Callable, Optional, Tuple, List

import numpy as np

from scipy.sparse.linalg import LinearOperator  # type: ignore

import lab


class MatVecOperator(LinearOperator):
    """Implementation of a generalized linear operator for matrix-vector
    products.

    Matrix-matrix products are implemented by naive iterative over columns.
    Note: use 'dot' to support composition of MatVecOperators.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        forward: Callable[[lab.Tensor], lab.Tensor],
        transpose: Optional[Callable[[lab.Tensor], lab.Tensor]] = None,
    ):
        """
        :param shape: the shape of the linear operator.
        :param forward: the function which computes matrix-vector products for the matrix.
        :param transpose: the function which computes transposed matrix products.
        """
        self.forward = forward
        self.transpose = transpose
        self.shape = shape
        try:
            self.dtype = np.dtype(lab.get_dtype())
        except:
            self.dtype = lab.get_dtype()

    def matvec(self, v: lab.Tensor) -> lab.Tensor:
        return self.forward(v)

    def _matvec(self, v: lab.Tensor):
        return self.forward(v)

    def rmatvec(self, v: lab.Tensor) -> lab.Tensor:
        if self.transpose is None:
            raise ValueError(
                "MatVec instance cannot compute transpose products because it was not provided with a transpose operator."
            )

        return self.transpose(v)

    def matmat(self, V: lab.Tensor) -> lab.Tensor:
        return lab.stack([self.matvec(col) for col in V.T], axis=1)

    def _rmatvec(self, v: lab.Tensor):
        if self.transpose is None:
            raise ValueError(
                "MatVec instance cannot compute transpose products because it was not provided with a transpose operator."
            )

        return self.transpose(v)

    def rmatmat(self, V: lab.Tensor) -> lab.Tensor:
        if self.transpose is None:
            raise ValueError(
                "MatVec instance cannot compute transpose products because it was not provided with a transpose operator."
            )

        return lab.stack([self.rmatvec(col) for col in V.T], axis=1)
