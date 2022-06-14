"""Utilities."""

from .linear_operators import MatVecOperator

from .linear import (
    solve_ne,
    lstsq_iterative_solve,
    linear_iterative_solve,
    get_preconditioner,
)

__all__ = [
    "solve_ne",
    "lstsq_iterative_solve",
    "linear_iterative_solve",
    "get_preconditioner",
]
