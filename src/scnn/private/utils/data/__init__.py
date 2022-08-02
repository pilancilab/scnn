"""Utilities for generating and processing datasets."""

from .synthetic import (
    gen_regression_data,
    gen_classification_data,
    gen_sparse_regression_problem,
    gen_sparse_nn_problem,
)
from .transforms import unitize_columns, train_test_split, add_bias_col


__all__ = [
    "gen_classification_data",
    "gen_regression_data",
    "gen_sparse_regression_problem",
    "add_bias_col",
    "unitize_columns",
    "train_test_split",
    "gen_sparse_nn_problem",
]
