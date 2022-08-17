"""Solution mappings for convex and non-convex formulations of neural network
training problems."""

from .mlps import grelu_solution_mapping, relu_solution_mapping

__all__ = [
    "grelu_solution_mapping",
    "relu_solution_mapping",
]
