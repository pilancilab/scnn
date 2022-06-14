"""Optimization procedure for iterative methods (e.g. gradient descent)."""

from logging import Logger, root, INFO
from typing import Dict, Any, List, Tuple, Callable, Optional

from tqdm.auto import tqdm  # type: ignore

import lab

from scnn.private.methods.optimization_procedures.optimization_procedure import (
    OptimizationProcedure,
    ITER_LOG_FREQ,
    METRIC_FREQ,
)

from scnn.private.methods.optimizers import Optimizer, ProximalOptimizer
from scnn.private.models.model import Model
from scnn.private.metrics import (
    update_metrics,
    init_metrics,
    format_recent_metrics,
    merge_metric_lists,
)
from scnn.private.methods.termination_criteria import TerminationCriterion


class IterativeOptimizationProcedure(OptimizationProcedure):

    """An iterative optimization procedure."""

    optimizer: Optimizer

    def __init__(
        self,
        optimizer: Optimizer,
        max_iters: int,
        term_criterion: TerminationCriterion,
        name: str = "",
        divergence_check: bool = True,
        batch_size: Optional[int] = None,
        log_freq: int = ITER_LOG_FREQ,
        metric_freq: int = METRIC_FREQ,
        pre_process: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """
        :param optimizer: an optimizer instance implementing the 'step' method.
        :param max_iters: the maximum number of iterations to run the optimizer.
        :param term_criterion: the criterion to use when checking for early termination.
        :param name: an optional name to display when printing the progress bar.
        :param divergence_check: an whether or not the optimization procedure should
            check for divergence behavior and terminate early.
        :param batch_size: the batch size to use when computing the objective.
            Defaults to `None' which indicates full-batch.
        :param log_freq: the frequency at which to information during optimization.
        :param metric_freq: the frequency at which to collect metrics, like objective,
            gradient-norm, etc, during optimization.
        :param pre_process: (optional) a function to call on the model after the it is
            initialized but *before* optimization starts.
        :param pre_process: (optional) a function to call on the model *after* optimization
            is complete.
        """
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.term_criterion = term_criterion

        self.name = name
        self.divergence_check = divergence_check
        self.batch_size = batch_size

        self.log_freq = log_freq
        self.metric_freq = metric_freq

        self.pre_process = pre_process
        self.post_process = post_process

    def reset(self):
        """Reset the optimization procedure (including any attached optimizers)
        to their original state."""
        self.optimizer.reset()

    def _pre_optimization(
        self,
        logger: Logger,
        model: Model,
        initializer: Callable[[Model], Model],
        train_data: Tuple[lab.Tensor, lab.Tensor],
        test_data: Tuple[lab.Tensor, lab.Tensor],
        metrics: Tuple[List[str], List[str], List[str]],
        final_metrics: Optional[Tuple[List[str], List[str], List[str]]],
        step_size: float,
    ) -> Tuple[Model, Dict[str, Any], Dict[str, Any], float, lab.Tensor]:

        exit_status: Dict[str, Any] = {}
        metrics_log: Dict[str, Any] = init_metrics(
            {}, merge_metric_lists(metrics, final_metrics)
        )
        X, y = train_data

        # initialize model
        start_time = self._get_start_time(None)
        model = initializer(model)

        # optional pre-processing.
        if self.pre_process is not None:
            model = self.pre_process(model, X, y)

        # compute initial objective and gradient.
        objective = model.objective(X, y, batch_size=self.batch_size)
        grad = model.grad(
            X,
            y,
            batch_size=self.batch_size,
            step_size=step_size,
        )

        # initial metrics
        metrics_log, _ = self._record_time(metrics_log, start_time)

        metrics_log = update_metrics(
            metrics_log,
            model,
            {"step_size": step_size},
            train_data,
            test_data,
            metrics,
            objective,
            grad,
            batch_size=self.batch_size,
        )
        logger.info(
            "Pre-Optimization Metrics: "
            + format_recent_metrics(metrics_log, metrics)
        )

        return (
            model,
            metrics_log,
            exit_status,
            objective,
            grad,
        )

    def _post_optimization(
        self,
        logger: Logger,
        model: Model,
        train_data: Tuple[lab.Tensor, lab.Tensor],
        test_data: Tuple[lab.Tensor, lab.Tensor],
        metrics: Tuple[List[str], List[str], List[str]],
        final_metrics: Optional[Tuple[List[str], List[str], List[str]]],
        exit_status: Dict[str, Any],
        metrics_log: Dict[str, Any],
        start_time: float,
        step_size: float,
    ) -> Tuple[Model, Dict[str, Any], Dict[str, Any]]:

        X, y = train_data

        start_time = self._get_start_time(start_time)

        # optional post-processing.
        if self.post_process is not None:
            model = self.post_process(model, X, y)

        metrics_log, start_time = self._record_time(metrics_log, start_time)

        # compute post-training metrics.
        objective = model.objective(X, y, batch_size=self.batch_size)
        grad = model.grad(X, y, batch_size=self.batch_size)

        metrics_log = update_metrics(
            metrics_log,
            model,
            {"step_size": step_size},
            train_data,
            test_data,
            merge_metric_lists(metrics, final_metrics),
            objective,
            grad,
            batch_size=self.batch_size,
        )

        logger.info(
            "Post-Optimization Metrics: "
            + format_recent_metrics(metrics_log, metrics)
        )

        return model, exit_status, metrics_log

    def __call__(
        self,
        logger: Logger,
        model: Model,
        initializer: Callable[[Model], Model],
        train_data: Tuple[lab.Tensor, lab.Tensor],
        test_data: Tuple[lab.Tensor, lab.Tensor],
        metrics: Tuple[List[str], List[str], List[str]],
        final_metrics: Optional[Tuple[List[str], List[str], List[str]]] = None,
        callback: Callable = None,
    ) -> Tuple[Dict[str, Any], Model, Dict[str, Any]]:
        """Optimize a model using an iterative optimize method by running a
        "optimization" loop. The loop runs for 'max_iters' number of
        iterations, collects optional metrics at every iteration, and will
        terminate early if the conditions of 'term_criterion' are met.

        :param logger: a logging.Logger object that can be used to log information during
            optimization.
        :param model: the model to optimize.
        :param initializer: a function that can be called to initialize the model.
        :param train_data: an (X, y) tuple representing the training set.
        :param test_data: an (X_test, y_test) tuple representing the test data.
        :param metrics: a tuple of the form (train_metrics, test_metrics, additional_metrics)
            specifying the metrics to be computed on the training set, test set, and data-independent
            metrics.
        :returns: (exit_status, model, metrics) --- execution information, the optimized model, and metrics from optimization.
        """

        # pre-optimization setup
        (
            model,
            metrics_log,
            exit_status,
            objective,
            grad,
        ) = self._pre_optimization(
            logger,
            model,
            initializer,
            train_data,
            test_data,
            metrics,
            final_metrics,
            self.optimizer.step_size,
        )
        (X, y), f0, start_time = train_data, objective, None

        # optimization loop
        verbose = root.level <= INFO
        for itr in tqdm(
            range(self.max_iters), desc=self.name, disable=(not verbose)
        ):
            start_time = self._get_start_time(start_time)
            if itr % self.log_freq == 0 and verbose:
                tqdm.write(format_recent_metrics(metrics_log, metrics))

            # check termination criteria
            if self.term_criterion(model, X, y, objective, grad):
                exit_status["success"] = True
                logger.info(
                    f"Termination criterion satisfied at iteration {itr}/{self.max_iters}. Exiting optimization loop."
                )
                break

            # check for divergence
            if self.divergence_check and objective > 1000 * f0:
                exit_status["success"] = False
                logger.warning(
                    f"Method diverged at {itr}/{self.max_iters}. Exiting optimization loop."
                )
                break

            model, objective, sp_exit_state = self.optimizer.step(
                model, X, y, objective, grad, batch_size=self.batch_size
            )

            if callback is not None:
                model = callback(model, X, y)

            # compute objective and gradient for next iteration.
            if objective is None:
                objective = model.objective(X, y, batch_size=self.batch_size)

            grad = model.grad(
                X,
                y,
                batch_size=self.batch_size,
                step_size=self.optimizer.step_size,
            )

            # calculate time and other metrics.
            if itr % self.metric_freq == 0:
                metrics_log, start_time = self._record_time(
                    metrics_log, start_time
                )
                metrics_log = update_metrics(
                    metrics_log,
                    model,
                    sp_exit_state,
                    train_data,
                    test_data,
                    metrics,
                    objective,
                    grad,
                    batch_size=self.batch_size,
                )

        # post-optimization clean-up.
        model, exit_status, metrics_log = self._post_optimization(
            logger,
            model,
            train_data,
            test_data,
            metrics,
            final_metrics,
            exit_status,
            metrics_log,
            start_time,
            self.optimizer.step_size,
        )

        # collect run-information
        if itr == self.max_iters - 1:
            exit_status["success"] = False
            logger.warning(
                "Max iterations reached before termination criterion was satisfied. Exiting optimization loop."
            )
        exit_status["iterations"] = itr + 1

        return exit_status, model, metrics_log
