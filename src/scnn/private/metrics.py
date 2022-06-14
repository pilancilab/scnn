"""Helpers for computing metrics during optimization.

TODO:
    - Active features should not take into account inactive neurons.
"""
import timeit
from typing import Any, Dict, List, Optional, Tuple

import scnn.private.loss_functions as loss_fns
import scnn.private.models.solution_mappings as sm
import lab
from scnn.private.models import AL_MLP, ConvexMLP
from scnn.private.models.model import Model


def as_list(x: Any) -> List[Any]:
    """Wrap argument into a list if it is not iterable.

    :param x: a (potential) singleton to wrap in a list.
    :returns: [x] if x is not iterable and x if it is.
    """
    # don't treat strings as iterables.
    if isinstance(x, str):
        return [x]

    try:
        _ = iter(x)
        return x
    except TypeError:
        return [x]


def format_recent_metrics(
    metric_dict: Dict[str, Any],
    metrics: Tuple[List[str], List[str], List[str]],
):
    """Print the most recent metric evaluation contained in the metric log.

    :param metric_dict: a dictionary used to record metric information during optimization. This dictionary should
        be initialize by calling 'init_metrics' before being passed to this function.
    :param metrics: a tuple of the form (train_metrics, test_metrics, additional_metrics)
        specifying the metrics to be computed on the training set, test set, and data-independent
        metrics.
    """
    train_metrics, test_metrics, additional_metrics = metrics
    metric_str = ""

    for metric_name in train_metrics:
        metric_str = (
            metric_str
            + f"Train Set {metric_name}: {metric_dict['train_' + metric_name][-1]}, "
        )

    for metric_name in test_metrics:
        metric_str = (
            metric_str
            + f"Test Set {metric_name}: {metric_dict['test_' + metric_name][-1]}, "
        )

    for metric_name in additional_metrics:
        if metric_name == "subproblem_metrics":
            continue

        metric_str = (
            metric_str + f"{metric_name}: {metric_dict[metric_name][-1]}, "
        )

    return metric_str


def merge_metric_lists(
    lm: Tuple[List[str], List[str], List[str]],
    rm: Optional[Tuple[List[str], List[str], List[str]]] = None,
) -> Tuple[List[str], List[str], List[str]]:

    if rm is None:
        return lm

    return (
        list(set(lm[0] + rm[0])),
        list(set(lm[1] + rm[1])),
        list(set(lm[2] + rm[2])),
    )


def init_metrics(
    metric_dict: Dict[str, List[Any]],
    metrics: Tuple[List[str], List[str], List[str]],
):
    """Initialize a metric dictionary with the necessary lists. This should be
    called before using 'update_metrics'.

    :param metric_dict: a (possibly empty) dictionary used to record metric information during optimization.
    :param metrics: a tuple of the form (train_metrics, test_metrics, additional_metrics)
        specifying the metrics to be computed on the training set, test set, and data-independent
        metrics.
    :returns: metric_dict with an empty list at each training/test metric key.
    """

    train_metrics, test_metrics, additional_metrics = metrics

    for metric_name in train_metrics:
        if "train_" + metric_name not in metric_dict:
            metric_dict["train_" + metric_name] = []

    # test metrics
    for metric_name in test_metrics:
        if "test_" + metric_name not in metric_dict:
            metric_dict["test_" + metric_name] = []

    for metric_name in additional_metrics:
        if metric_name not in metric_dict:
            metric_dict[metric_name] = []

    return metric_dict


def update_metrics(
    metric_dict: Dict[str, List[Any]],
    model: Model,
    sp_exit_state: Dict[str, Any],
    train_data: Tuple[lab.Tensor, lab.Tensor],
    test_data: Tuple[lab.Tensor, lab.Tensor],
    metrics: Tuple[List[str], List[str], List[str]],
    train_objective: Optional[lab.Tensor] = None,
    train_grad: Optional[lab.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """Update a dictionary of metrics given a model, a training set, a test
    set, and a list of metrics to use.

    :param metric_dict: a dictionary used to record metric information during optimization. This dictionary should
        be initialize by calling 'init_metrics' before being passed to this function.
    :param model: the model for which to compute the metrics.
    :param sp_exit_state: the exit status of any sub-processes called by the optimizer.
    :param train_data: an (X,y) tuple containing the training dataset.
    :param test_data: an (X_test, y_test) tuple containing the test/validation set.
    :param metrics: a tuple of the form (train_metrics, test_metrics, additional_metrics)
        specifying the metrics to be computed on the training set, test set, and data-independent
        metrics.
    :param batch_size: the batch size to use when computing the objective.
        Defaults to `None' which indicates full-batch.
    :returns: metric_dict with an empty list at each training/test metric key.
    """

    train_metrics, test_metrics, additional_metrics = metrics

    # training metrics
    for metric_name in train_metrics:
        metric_dict = compute_metric(
            metric_name,
            metric_dict,
            "train_" + metric_name,
            model,
            sp_exit_state,
            train_data,
            train_objective,
            train_grad,
            batch_size=batch_size,
        )

    # test metrics
    model.eval()
    for metric_name in test_metrics:
        metric_dict = compute_metric(
            metric_name,
            metric_dict,
            "test_" + metric_name,
            model,
            sp_exit_state,
            test_data,
            batch_size=batch_size,
        )
    model.train()

    for metric_name in additional_metrics:
        metric_dict = compute_metric(
            metric_name,
            metric_dict,
            metric_name,
            model,
            sp_exit_state,
            train_data,
            batch_size=batch_size,
        )

    return metric_dict


def compute_metric(
    metric_name: str,
    metric_dict: Dict[str, Any],
    dict_key: str,
    model: Model,
    sp_exit_state: Dict[str, Any],
    data: Tuple[lab.Tensor, lab.Tensor],
    objective: Optional[lab.Tensor] = None,
    grad: Optional[lab.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Any:
    """Lookup a metric by name and compute it for a given model and dataset.

    :param metric_name: the name of the metric to compute.
    :param metric_dict: a dictionary used to record metric information during optimization. This dictionary should
        be initialize by calling 'init_metrics' before being passed to this function.
    :param dict_key: the key at which the metric should be stored in 'metric_dict'.
    :param model: the model for which the metric should be computed.
    :param sp_exit_state: the exit status of any sub-processes called by the optimizer.
    :param data: an (X,y) tuple containing a training/validation/test set on which to compute the metric.
    :param batch_size: the batch size to use when computing the objective.
        Defaults to `None' which indicates full-batch.
    :returns: the value of the desired metric.
    """
    X, y = data
    metric = None

    if metric_name == "objective":
        if objective is None:
            objective = model.objective(X, y, batch_size=batch_size)
        metric = lab.to_scalar(objective)
    elif metric_name == "base_objective":
        objective = model.objective(
            X, y, batch_size=batch_size, ignore_lagrange_penalty=True
        )
        metric = lab.to_scalar(objective)
    elif metric_name == "squared_error":
        metric = lab.to_scalar(
            loss_fns.squared_error(model(X, batch_size=batch_size), y)
            / y.shape[0]
        )
    elif metric_name == "grad_norm":
        if grad is None:
            grad = model.grad(X, y, batch_size=batch_size)
        metric = lab.to_scalar(lab.sum(grad ** 2))
    elif metric_name == "binary_accuracy":
        metric = lab.to_scalar(
            lab.sum(lab.sign(model(X, batch_size=batch_size)) == y)
            / y.shape[0]
        )
    elif metric_name == "binned_accuracy":
        metric = lab.to_scalar(
            loss_fns.binned_accuracy(model(X, batch_size=batch_size), y, 10)
        )
    elif metric_name == "accuracy":
        if y.shape[1] > 1:
            metric = lab.to_scalar(
                loss_fns.accuracy(model(X, batch_size=batch_size), y)
            )
        else:
            metric = lab.to_scalar(
                lab.sum(lab.sign(model(X, batch_size=batch_size)) == y)
                / y.shape[0]
            )
    elif metric_name == "time_stamp":
        metric = timeit.default_timer()
    elif metric_name == "total_neurons":
        metric = model.p
    elif metric_name == "feature_sparsity":
        reduced_weights = model.weights
        feature_weights = lab.sum(
            lab.abs(reduced_weights),
            axis=tuple(range(len(reduced_weights.shape) - 1)),
        )
        # feature_weights = feature_weights + lab.sum(lab.abs(model.U), axis=-1)

        metric = lab.to_scalar(
            lab.sum(feature_weights == 0.0) / reduced_weights.shape[-1]
        )
    elif metric_name == "active_features":
        reduced_weights = model.weights
        feature_weights = lab.sum(
            lab.abs(reduced_weights),
            axis=tuple(range(len(reduced_weights.shape) - 1)),
        )
        # feature_weights = feature_weights + lab.sum(lab.abs(model.U), axis=-1)
        metric = lab.to_scalar(lab.sum(feature_weights != 0.0))
    elif metric_name == "group_sparsity":
        metric = lab.to_scalar(
            lab.sum(lab.sum(model.weights, axis=-1) == 0.0)
            / (lab.size(model.weights) / model.weights.shape[-1])
        )
    elif metric_name == "active_neurons":
        metric = lab.to_scalar(lab.sum(lab.sum(model.weights, axis=-1) != 0.0))
    elif metric_name == "sparsity":
        metric = lab.to_scalar(
            lab.sum(model.weights == 0) / lab.size(model.weights)
        )
    elif metric_name == "group_norms":
        metric = lab.to_scalar(
            lab.sum(lab.sqrt(lab.sum(model.weights ** 2, axis=-1)))
        )
    elif metric_name == "dual_param_norm":
        try:
            metric = lab.to_scalar(lab.sum(model.i_multipliers ** 2))
        except:
            metric = lab.tensor([0.0])
    elif metric_name == "constraint_gaps":
        try:
            e_gap, i_gap = model.constraint_gaps(X)
        except:
            e_gap = i_gap = lab.tensor([0.0])

        metric = lab.to_scalar(lab.sum(e_gap ** 2 + lab.smax(i_gap, 0) ** 2))
    elif metric_name == "lagrangian_grad":
        try:
            metric = lab.to_scalar(lab.sum(model.lagrangian_grad(X, y) ** 2))
        except:
            metric = 0.0
    elif metric_name == "num_backtracks":
        metric = sp_exit_state.get("attempts", 1) - 1
    elif metric_name == "sp_success":
        metric = sp_exit_state.get("success", True)
    elif metric_name == "step_size":
        metric = lab.to_scalar(sp_exit_state.get("step_size", 0.0))
    elif metric_name == "subproblem_metrics":
        for key, value in sp_exit_state.get("subproblem_metrics", {}).items():
            old_value = metric_dict.get(f"subproblem_{key}", [])
            metric_dict[f"subproblem_{key}"] = old_value + as_list(value)
    elif "nc_" in metric_name:
        nc_model = model

        # compute the non-convex model if one exists
        if isinstance(model, ConvexMLP):
            nc_model = sm.get_nc_formulation(model, remove_sparse=True)

        compute_metric(
            metric_name.split("nc_")[1],
            metric_dict,
            dict_key,
            nc_model,
            sp_exit_state,
            data,
            None,
            None,
            batch_size,
        )

    else:
        raise ValueError(
            f"Metric {metric_name} not recognized! Please register it in 'metrics.py'"
        )
    if metric is not None:
        metric_dict.get(dict_key, []).append(metric)

    return metric_dict
