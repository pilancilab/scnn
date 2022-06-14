"""Base class for optimization procedures."""

from logging import Logger
from typing import Dict, Any, List, Tuple, Callable, Optional
import timeit

import lab

from scnn.private.methods.optimizers import Optimizer
from scnn.private.models.model import Model
from scnn.private.metrics import (
    update_metrics,
    init_metrics,
    format_recent_metrics,
    merge_metric_lists,
)
from scnn.private.methods.termination_criteria import TerminationCriterion

# CONSTANTS

ITER_LOG_FREQ = 25
EPOCH_LOG_FREQ = 5

METRIC_FREQ = 1

# CLASSES


class OptimizationProcedure:

    """A single-call optimization procedure."""

    term_criterion: TerminationCriterion

    def __init__(
        self,
        optimizer: Optimizer,
        pre_process: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """
        :param optimizer: an callable optimizer instance.
        """

        self.optimizer = optimizer
        self.pre_process = pre_process
        self.post_process = post_process

    def reset(self):
        """Reset the optimization procedure (including any attached optimizers)
        to their original state."""
        pass

    def _record_time(self, metrics_log, start_time):
        """Record time elapsed since start_time."""
        metrics_log["time"] = metrics_log.get("time", []) + [
            timeit.default_timer() - start_time
        ]

        return metrics_log, None

    def _get_start_time(self, start_time):
        return timeit.default_timer() if start_time is None else start_time

    def __call__(
        self,
        logger: Logger,
        model: Model,
        initializer: Callable[[Model], Model],
        train_data: Tuple[lab.Tensor, lab.Tensor],
        test_data: Tuple[lab.Tensor, lab.Tensor],
        metrics: Tuple[List[str], List[str], List[str]],
        final_metrics: Optional[Tuple[List[str], List[str], List[str]]] = None,
    ) -> Tuple[Dict[str, Any], Model, Dict[str, Any]]:
        """Optimize a model using an optimize method by directly calling an
        optimization procedure. Training and test metrics are collected before
        and after execution.

        :param logger: a logging.Logger object that can be used to log information during
            optimization.
        :param model: the model to optimize.
        :param initializer: a function that can be called to initialize the model.
        :param train_data: an (X, y) tuple representing the training set.
        :param test_data: an (X_test, y_test) tuple representing the test data.
        :param train_metrics: a list of strings indicating which metrics to evaluate on the
            training set.
        :param test_metrics: a list of strings indicating which metrics to evaluate on the
            test set.
        :param additional_metrics: a list of strings identifying the further metrics that should be computed.
        :returns: (exit_status, model, metrics) --- execution information, the optimized model, and metrics from optimization.
        """
        # setup
        X, y = train_data
        metrics_log: Dict[str, Any] = init_metrics(
            {}, merge_metric_lists(metrics, final_metrics)
        )

        start_time = self._get_start_time(None)
        model = initializer(model)

        # optional pre-processing.
        if self.pre_process is not None:
            model = self.pre_process(model, X, y)

        metrics_log, start_time = self._record_time(metrics_log, start_time)

        # pre-training metrics
        metrics_log = update_metrics(
            metrics_log, model, {}, train_data, test_data, metrics
        )

        logger.info(
            "Pre-Training Metrics: " + format_recent_metrics(metrics_log, metrics)
        )

        start_time = self._get_start_time(start_time)
        model, exit_status = self.optimizer(
            model,
            X,
            y,
        )

        # optional post-processing.
        if self.post_process is not None:
            model = self.post_process(model, X, y)

        # post-training metrics
        metrics_log, start_time = self._record_time(metrics_log, start_time)
        metrics_log = update_metrics(
            metrics_log,
            model,
            exit_status,
            train_data,
            test_data,
            merge_metric_lists(metrics, final_metrics),
        )

        logger.info(
            "Post-Training Metrics: " + format_recent_metrics(metrics_log, metrics)
        )

        return exit_status, model, metrics_log
