"""Convert metrics from :module:`scnn.metrics` into internal
representations."""
from typing import Tuple, List, Dict, Any
from copy import deepcopy

import numpy as np

# Public Facing Objects

from scnn.metrics import Metrics


def build_metrics_tuple(
    metrics: Metrics,
) -> Tuple[List[str], List[str], List[str]]:
    """Convert Metrics instance into tuple of metrics for internal use.

    Args:
        metrics: object specifying which metrics should be collected during optimization.

    Returns:
        (train_metrics, test_metrics, additional_metrics) --- tuple of list of strings specifying which metrics should be collected.
    """
    train_metrics = []
    test_metrics = []
    additional_metrics = []

    for key, value in metrics.metrics_to_collect.items():
        if not value:
            continue

        if key == "objective":
            train_metrics.append("objective")
        elif key == "grad_norm":
            train_metrics.append("grad_norm")
        elif key == "time":
            continue
        elif key == "model_loss":
            train_metrics.append("base_objective")
        elif key == "constraint_gaps":
            train_metrics.append("constraint_gaps")
        elif key == "lagrangian_grad":
            train_metrics.append("lagrangian_grad")
        # use convex model for training (same as non-convex)
        elif key == "train_accuracy":
            train_metrics.append("accuracy")
        # use convex model for training (same as non-convex)
        elif key == "train_mse":
            train_metrics.append("squared_error")
        # use non-convex model for testing
        elif key == "test_accuracy":
            test_metrics.append("nc_accuracy")
        # use non-convex model for testing
        elif key == "test_mse":
            test_metrics.append("nc_squared_error")
        elif key == "active_neurons":
            additional_metrics.append("active_neurons")
        elif key == "neuron_sparsity":
            additional_metrics.append("group_sparsity")
        elif key == "active_features":
            additional_metrics.append("active_features")
        elif key == "feature_sparsity":
            additional_metrics.append("feature_sparsity")
        elif key == "active_weights":
            additional_metrics.append("active_weights")
        elif key == "weight_sparsity":
            additional_metrics.append("weight_sparsity")

    return train_metrics, test_metrics, additional_metrics


def update_public_metrics(
    metrics: Metrics, internal_metrics: Dict[str, Any]
) -> Metrics:
    """Update public-facing metrics object with optimization results.

    Args:
        metrics: a dictionary mapping metric names to data collected during optimization.
        internal_metrics: a public-facing metrics object.

    Returns:
        A new public-facing metrics object updated with metrics collected during optimization.
    """
    metrics = deepcopy(metrics)

    for key, value in internal_metrics.items():
        if key == "train_objective":
            metrics.objective = np.array(value)
        elif key == "train_grad_norm":
            metrics.grad_norm = np.array(value)
        elif key == "time":
            metrics.time = np.cumsum(np.array(value))
        elif key == "train_base_objective":
            metrics.model_loss = value
        elif key == "train_constraint_gaps":
            metrics.constraint_gaps = value
        elif key == "train_lagrangian_grad":
            metrics.lagrangian_grad = value
        elif key == "train_accuracy":
            metrics.train_accuracy = value
        # use convex model for training (same as non-convex)
        elif key == "train_squared_error":
            metrics.train_mse = value
        # use non-convex model for testing
        elif key == "test_nc_accuracy":
            metrics.test_accuracy = value
        # use non-convex model for testing
        elif key == "test_nc_squared_error":
            metrics.test_mse = value
        elif key == "active_neurons":
            metrics.active_neurons = value
        elif key == "group_sparsity":
            metrics.neuron_sparsity = value
        elif key == "active_features":
            metrics.active_features = value
        elif key == "feature_sparsity":
            metrics.feature_sparsity = value
        elif key == "active_weights":
            metrics.active_weights = value
        elif key == "weight_sparsity":
            metrics.weight_sparsity = value

    return metrics
