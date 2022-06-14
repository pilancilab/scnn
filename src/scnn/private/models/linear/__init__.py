"""Convex re-formulations of neural network models."""

from .l2_regression import LinearRegression
from .logistic_regression import LogisticRegression

__all__ = [
    "LinearRegression",
    "LogisticRegression",
]
