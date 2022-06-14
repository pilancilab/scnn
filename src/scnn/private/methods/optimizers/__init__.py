"""Optimizers.

TODO:
    - Add optimal gradient method for L2-regularized and unregularized problems.
"""

# ===== module exports ===== #

from .optimizer import Optimizer
from .proximal_optimizer import ProximalOptimizer, ProximalLSOptimizer
from .meta_optimizer import MetaOptimizer
from .gd import GD, GDLS
from .pgd import PGD, PGDLS
from .fista import FISTA
from .augmented_lagrangian import AugmentedLagrangian

__all__ = [
    "Optimizer",
    "ProximalOptimizer",
    "ProximalLSOptimizer",
    "MetaOptimizer",
    "GD",
    "GDLS",
    "PGD",
    "PGDLS",
    "FISTA",
    "AugmentedLagrangian",
]
