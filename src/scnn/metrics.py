"""Metrics that can be recorded while training models."""

from typing import Dict

import numpy as np


class Metrics(object):
    """Metrics collected while optimizing models.

    By default, only the objective, training time, and norm of the
    minimum-norm subgradient are collected.

    Attributes:
        objective: the optimization objective, including any constraint
            penalty terms.
        time: the cumulative wall-clock time.
        grad_norm: squared 2-norm of the minimum-norm subgradient of the
            optimization objective, including penalty terms.
        model_loss: the regularized loss of the model on the training set. This
            is the same as `objective` but does not include penalties.
        constraint_gaps: 2-norm of constraint violations.
            Ignored for unconstrained problems.
        lagrangian_grad: 2-norm of the (primal) subgradient of the
            Lagrangian function. Ignored for unconstrained problems.
        train_accuracy: accuracy on the training set.
        train_mse: mean squared error on the training set.
        test_accuracy": accuracy on the test set.
        test_mse: means squared error on the test set.
        total_neurons: total number of neurons (active and inactive) in the
            model.
        neuron_sparsity: proportion of neurons which are not used by the model
          (ie. all weights are exactly zero for those neurons).
        active_neurons: the number of neurons which are active (ie. the weights
            for those neurons are not exactly zero).
        total_features: total number of features in the dataset (both used and
            not used by the model).
        feature_sparsity: proportion of features which are not used by the
            model (ie. all weights are exactly zero for those features).
        active_features: the number of features currently used by the model.
        weight_sparsity: proportion of weights are which zero.
        weight_sparsity: the R^2 coefficient, also called the coefficient
            of determination.
        r_squared: the R^2 metric, also called the coefficient of
            determination.
        auc: area under the ROC curve.
        metric_freq: the frequency (in iterations) at which metrics should be
            collected in iterations.
        metrics_to_collect: internal dictionary specifying which metrics
            should be collected.
    """

    test_metrics = False

    objective: np.ndarray
    grad_norm: np.ndarray
    time: np.ndarray
    model_loss: np.ndarray
    constraint_gaps: np.ndarray
    lagrangian_grad: np.ndarray
    train_accuracy: np.ndarray
    train_mse: np.ndarray
    test_accuracy: np.ndarray
    test_mse: np.ndarray
    total_neurons: np.ndarray
    neuron_sparsity: np.ndarray
    active_neurons: np.ndarray
    total_features: np.ndarray
    feature_sparsity: np.ndarray
    active_features: np.ndarray
    active_weights: np.ndarray
    weight_sparsity: np.ndarray
    r_squared: np.ndarray
    auc: np.ndarray

    def __init__(
        self,
        metric_freq: int = 25,
        objective: bool = True,
        grad_norm: bool = True,
        time: bool = True,
        model_loss: bool = False,
        constraint_gaps: bool = False,
        lagrangian_grad: bool = False,
        train_accuracy: bool = False,
        train_mse: bool = False,
        test_accuracy: bool = False,
        test_mse: bool = False,
        total_neurons: bool = False,
        neuron_sparsity: bool = False,
        active_neurons: bool = False,
        total_features: bool = False,
        feature_sparsity: bool = False,
        active_features: bool = False,
        active_weights: bool = False,
        weight_sparsity: bool = False,
        r_squared: bool = False,
        auc: bool = False,
    ):
        """
        Args:
            metric_freq: the frequency at which to log metrics.
            objective: the optimization objective, including any constraint
                penalty terms.
            time: the cumulative wall-clock time.
            grad_norm: squared 2-norm of the minimum-norm subgradient of the
                optimization objective, including penalty terms.
            model_loss: the regularized loss of the model on the training set.
                This is the same as `objective` but does not include penalties.
            constraint_gaps: 2-norm of constraint violations.
                Ignored for unconstrained problems.
            lagrangian_grad: 2-norm of the (primal) subgradient of the
                Lagrangian function. Ignored for unconstrained problems.
            train_accuracy: accuracy on the training set.
            train_mse: mean squared error on the training set.
            test_accuracy": accuracy on the test set.
            test_mse: means squared error on the test set.
            total_neurons: total number of neurons in the model.
            neuron_sparsity: proportion of neurons which are not used by the
                model (ie. all weights are exactly zero for those neurons).
            active_neurons: the number of neurons which are active (ie. the
                weights for those neurons are not exactly zero).
            total_features: total number of features in the dataset (both used
                and not used by the model).
            feature_sparsity: proportion of features which are not used by the
                model (ie. all weights are exactly zero for those features).
            active_features: number of features used by the model.
            active_features: number of features used by the model.
            weight_sparsity: proportion of weights are which zero.
            r_squared: the R^2 metric, also called the coefficient of
                determination.
            auc: area under the ROC curve.
        """

        self.metric_freq = metric_freq
        self.metrics_to_collect = {
            "objective": objective,
            "grad_norm": grad_norm,
            "time": time,
            "model_loss": model_loss,
            "constraint_gaps": constraint_gaps,
            "lagrangian_grad": lagrangian_grad,
            "train_accuracy": train_accuracy,
            "train_mse": train_mse,
            "test_accuracy": test_accuracy,
            "test_mse": test_mse,
            "total_neurons": total_neurons,
            "neuron_sparsity": neuron_sparsity,
            "active_neurons": active_neurons,
            "total_features": total_features,
            "feature_sparsity": feature_sparsity,
            "active_features": active_features,
            "active_weights": active_weights,
            "weight_sparsity": weight_sparsity,
            "r_squared": r_squared,
            "auc": auc,
        }

        if test_mse or test_accuracy:
            self.test_metrics = True

    def has_test_metrics(self) -> bool:
        """Returns `True` if any test-set metric is enabled, `False`
        otherwise."""
        return self.test_metrics

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert metrics object into a dictionary."""

        metrics = {}
        if self.metrics_to_collect["objective"]:
            metrics["objective"] = self.objective
        if self.metrics_to_collect["grad_norm"]:
            metrics["grad_norm"] = self.grad_norm
        if self.metrics_to_collect["time"]:
            metrics["time"] = self.time
        if self.metrics_to_collect["model_loss"]:
            metrics["model_loss"] = self.model_loss
        if self.metrics_to_collect["constraint_gaps"]:
            metrics["constraint_gaps"] = self.constraint_gaps
        if self.metrics_to_collect["lagrangian_grad"]:
            metrics["lagrangian_grad"] = self.lagrangian_grad
        if self.metrics_to_collect["train_accuracy"]:
            metrics["train_accuracy"] = self.train_accuracy
        if self.metrics_to_collect["train_mse"]:
            metrics["train_mse"] = self.train_mse
        if self.metrics_to_collect["test_accuracy"]:
            metrics["test_accuracy"] = self.test_accuracy
        if self.metrics_to_collect["test_mse"]:
            metrics["test_mse"] = self.test_mse
        if self.metrics_to_collect["total_neurons"]:
            metrics["total_neurons"] = self.total_neurons
        if self.metrics_to_collect["neuron_sparsity"]:
            metrics["neuron_sparsity"] = self.neuron_sparsity
        if self.metrics_to_collect["active_neurons"]:
            metrics["active_neurons"] = self.active_neurons
        if self.metrics_to_collect["total_features"]:
            metrics["total_features"] = self.total_features
        if self.metrics_to_collect["feature_sparsity"]:
            metrics["feature_sparsity"] = self.feature_sparsity
        if self.metrics_to_collect["active_features"]:
            metrics["active_features"] = self.active_features
        if self.metrics_to_collect["active_weights"]:
            metrics["active_weights"] = self.active_weights
        if self.metrics_to_collect["weight_sparsity"]:
            metrics["weight_sparsity"] = self.weight_sparsity
        if self.metrics_to_collect["r_squared"]:
            metrics["r_squared"] = self.r_squared
        if self.metrics_to_collect["auc"]:
            metrics["auc"] = self.auc

        return metrics
