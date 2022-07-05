"""Optimize neural networks using convex reformulations.

TODO:
    - extract types into a new `types.py` module.

    - use_bias: update mapping for convex/non-convex models to support bias
        terms without data augmentation.

"""
import math
import os
import pickle as pkl
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np

from scnn.activations import sample_gate_vectors
from scnn.metrics import Metrics
from scnn.models import ConvexGatedReLU, ConvexReLU, Model
from scnn.private.interface import (
    build_internal_model,
    build_internal_regularizer,
    build_metrics_tuple,
    build_optimizer,
    build_public_model,
    get_logger,
    normalized_into_input_space,
    process_data,
    set_device,
    update_public_metrics,
    update_public_model,
)
from scnn.private.models.solution_mappings import get_nc_formulation
from scnn.regularizers import Regularizer
from scnn.solvers import (
    AL,
    RFISTA,
    ApproximateConeDecomposition,
    Optimizer,
)

# Types

Formulation = Literal["gated_relu", "relu"]
Device = Literal["cpu", "cuda"]
Dtype = Literal["float32", "float64"]


def optimize(
    formulation: Formulation,
    max_neurons: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    regularizer: Optional[Regularizer] = None,
    bias: bool = False,
    return_convex: bool = False,
    unitize_data: bool = True,
    verbose: bool = False,
    log_file: str = None,
    device: Device = "cpu",
    dtype: Dtype = "float32",
    seed: int = 778,
) -> Tuple[Model, Metrics]:
    """Convenience function for training neural networks by convex
    reformulation.

    Args:
        formulation: the convex reformulation to solve. Must be one of

            - `"gated_relu"`: train a network with Gated ReLU activations.

            - `"relu"`: train a network with ReLU activations.

        max_neurons: the maximum number of neurons in the convex reformulation.
        X_train: an :math:`n \\times d` matrix of training examples.
        y_train: an :math:`n \\times c` or vector matrix of training targets.
        X_test: an :math:`m \\times d` matrix of test examples.
        y_test: an :math:`n \\times c` or vector matrix of test targets.
        regularizer: an optional regularizer for the convex reformulation.
            Defaults to no regularization.
        bias: whether or not to use a bias in the model.
        return_convex: whether or not to return the convex reformulation
            instead of the final non-convex model.
        unitize_data: whether or not to unitize the column norms of the
            training set. This can improve conditioning during optimization.
        verbose: whether or not the solver should print verbosely during
            optimization.
        log_file: a path to an optional log file.
        device: the device on which to run. Must be one of `cpu` (run on CPU)
            or `cuda` (run on cuda-enabled GPUs if available).
        dtype: the floating-point type to use. `"float32"` is faster than
            `"float64"` but can lead to excessive numerical errors on badly
            conditioned datasets.
        seed: an integer seed for reproducibility.

    Returns:
        (Model, Metrics): the optimized model and metrics collected during
        optimization.
    """

    model: Model
    solver: Optimizer

    d = len(X_train[0])
    # check number of outputs
    try:
        c = len(y_train[0])
    except TypeError:
        c = 1

    # Instantiate convex model and other options.
    if formulation == "gated_relu":
        G = sample_gate_vectors(seed, d, max_neurons)
        model = ConvexGatedReLU(G, c=c, bias=bias)
        solver = RFISTA(model)
    elif formulation == "relu":
        # ReLU models can have up to 2 * G neurons.
        G = sample_gate_vectors(seed, d, math.floor(max_neurons / 2))
        model = ConvexReLU(G, c=c, bias=bias)
        solver = AL(model)
    else:
        raise ValueError(f"Convex formulation {formulation} not recognized!")

    metrics = Metrics()

    return optimize_model(
        model,
        solver,
        metrics,
        X_train,
        y_train,
        X_test,
        y_test,
        regularizer,
        return_convex,
        unitize_data,
        verbose,
        log_file,
        device,
        dtype,
        seed,
    )


def optimize_model(
    model: Model,
    solver: Optimizer,
    metrics: Metrics,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    regularizer: Optional[Regularizer] = None,
    return_convex: bool = False,
    unitize_data: bool = True,
    verbose: bool = False,
    log_file: str = None,
    device: Device = "cpu",
    dtype: Dtype = "float32",
    seed: int = 778,
) -> Tuple[Model, Metrics]:
    """Train a neural network by convex reformulation.

    Args:
        model: a convex reformulation of a neural network model.
        solver: the optimizer to use when solving the reformulation.
        metrics: a object specifying which metrics to collect during
            optimization.
        X_train: an :math:`n \\times d` matrix of training examples.
        y_train: an :math:`n \\times c` or vector matrix of training targets.
        X_test: an :math:`m \\times d` matrix of test examples.
        y_test: an :math:`n \\times c` or vector matrix of test targets.
        regularizer: an optional regularizer for the convex reformulation.
        return_convex: whether or not to return the convex reformulation
            instead of the final non-convex model.
        unitize_data: whether or not to unitize the column norms of the
            training set. This can improve conditioning during optimization.
        verbose: whether or not the solver should print verbosely during
            optimization.
        log_file: a path to an optional log file.
        device: the device on which to run. Must be one of `cpu` (run on CPU)
            or `cuda` (run on cuda-enabled GPUs if available).
        dtype: the floating-point type to use. `"float32"` is faster than
            `"float64"` but can lead to excessive numerical errors on badly
            conditioned datasets.
        seed: an integer seed for reproducibility.

    Returns:
        The optimized model and metrics collected during optimization.
    """
    logger = get_logger("scnn", verbose, False, log_file)

    # set backend settings.
    if solver.cpu_only() and device != "cpu":
        logger.warning(
            f"Solver {solver} only supports CPU. User supplied device {device}\
            has been overridden."
        )
        device = "cpu"

    set_device(device, dtype, seed)

    if metrics.has_test_metrics() and (X_test is None or y_test is None):
        logger.warning(
            "Metrics specifies test metrics, but no test set was provided. \
            Test metrics will be collected on the training set. \n"
        )

    # Note: this unitizes columns of data matrix.
    (X_train, y_train), (X_test, y_test), column_norms = process_data(
        X_train,
        y_train,
        X_test,
        y_test,
        unitize_data,
        model.bias,
    )

    internal_model = build_internal_model(model, regularizer, X_train)
    opt_procedure = build_optimizer(solver, regularizer, metrics)
    metrics_tuple = build_metrics_tuple(metrics)

    initializer = lambda model: model

    exit_status, internal_model, internal_metrics = opt_procedure(
        logger,
        internal_model,
        initializer,
        (X_train, y_train),
        (X_test, y_test),
        metrics_tuple,
    )

    metrics = update_public_metrics(metrics, internal_metrics)

    # convert internal metrics

    # transform model back to original data space if necessary.
    if unitize_data:
        internal_model.weights = normalized_into_input_space(
            internal_model.weights, column_norms
        )

    if return_convex:
        return update_public_model(model, internal_model), metrics

    # convert into internal non-convex model
    nc_internal_model = get_nc_formulation(internal_model, remove_sparse=True)

    # create non-convex model
    return build_public_model(nc_internal_model, model.bias), metrics


def optimize_path(
    model: Model,
    solver: Optimizer,
    path: List[Regularizer],
    metrics: Metrics,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    warm_start: bool = True,
    save_path: Optional[str] = None,
    return_convex: bool = False,
    unitize_data: bool = True,
    verbose: bool = False,
    log_file: str = None,
    device: Device = "cpu",
    dtype: Dtype = "float32",
    seed: int = 778,
) -> Tuple[List[Union[Model, str]], List[Metrics]]:
    """Train a neural network by convex reformulation.

    Args:
        model: a convex reformulation of a neural network model.
        solver: the optimizer to use when solving the reformulation.
        path: a list of regularizer objects specifying the regularization path
            to explore.
        metrics: a object specifying which metrics to collect during
            optimization.
        X_train: an :math:`n \\times d` matrix of training examples.
        y_train: an :math:`n \\times c` or vector matrix of training targets.
        X_test: an :math:`m \\times d` matrix of test examples.
        y_test: an :math:`n \\times c` or vector matrix of test targets.
        warm_start: whether or not to warm-start each optimization problem in
            the path using the previous solution.
        save_path: string specifying a directory where models in the
            regularization path should be saved. All models will be retained
            in memory and returned if `save_path = None`.
        return_convex: whether or not to return the convex reformulation
            instead of the final non-convex model.
        unitize_data: whether or not to unitize the column norms of the
            training set. This can improve conditioning during optimization.
        verbose: whether or not the solver should print verbosely during
            optimization.
        log_file: a path to an optional log file.
        device: the device on which to run. Must be one of `cpu` (run on CPU)
            or `cuda` (run on cuda-enabled GPUs if available).
        dtype: the floating-point type to use. `"float32"` is faster than
            `"float64"` but can lead to excessive numerical errors on badly
            conditioned datasets.
        seed: an integer seed for reproducibility.
    """
    # set backend settings.
    logger = get_logger("scnn", verbose, False, log_file)

    # set backend settings.
    if solver.cpu_only() and device != "cpu":
        logger.warning(
            f"Solver {solver} only supports CPU. User supplied device {device}\
              has been overridden."
        )
        device = "cpu"

    set_device(device, dtype, seed)

    if metrics.has_test_metrics() and (X_test is None or y_test is None):
        logger.warning(
            "Metrics specifies test metrics, but no test set was provided.\
            Test metrics will be collected on the training set. \n"
        )

    # Note: this unitizes columns of data matrix.
    (X_train, y_train), (X_test, y_test), column_norms = process_data(
        X_train,
        y_train,
        X_test,
        y_test,
        unitize_data,
    )

    internal_model = build_internal_model(model, path[0], X_train)
    initializer = lambda model: model

    metrics_list: List[Metrics] = []
    model_list: List[Union[Model, str]] = []

    for regularizer in path:
        # update internal regularizer
        internal_model.regularizer = build_internal_regularizer(regularizer)
        opt_procedure = build_optimizer(solver, regularizer, metrics)

        metrics_tuple = build_metrics_tuple(metrics)

        exit_status, internal_model, internal_metrics = opt_procedure(
            logger,
            internal_model,
            initializer,
            (X_train, y_train),
            (X_test, y_test),
            metrics_tuple,
        )

        # regularizer to_string to generate path
        metrics = update_public_metrics(metrics, internal_metrics)
        cur_weights = internal_model.weights
        # transform model back to original data space if necessary.
        if unitize_data:
            internal_model.weights = normalized_into_input_space(
                internal_model.weights, column_norms
            )

        if return_convex:
            model_to_save = update_public_model(model, internal_model)
        else:
            nc_internal_model = get_nc_formulation(internal_model, remove_sparse=True)
            model_to_save = build_public_model(nc_internal_model, model.bias)

        if save_path is not None:

            reg_path = os.path.join(save_path, str(regularizer))
            os.makedirs(save_path, exist_ok=True)
            with open(reg_path, "wb") as f:
                pkl.dump((model_to_save, metrics), f)
            model_list.append(reg_path)

        else:
            model_list.append(deepcopy(model_to_save))

        metrics_list.append(deepcopy(metrics))
        internal_model.weights = cur_weights

    return model_list, metrics_list
