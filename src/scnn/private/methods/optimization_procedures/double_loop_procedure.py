"""Optimization procedure for iterative methods (e.g. gradient descent)."""
from logging import Logger, root, INFO
from typing import Dict, Any, List, Tuple, Callable, Optional
import timeit

from tqdm.auto import tqdm  # type: ignore

import lab

from scnn.private.methods.optimization_procedures.optimization_procedure import (
    ITER_LOG_FREQ,
    METRIC_FREQ,
)
from scnn.private.methods.optimization_procedures.iterative import (
    IterativeOptimizationProcedure,
)

from scnn.private.methods.optimizers import (
    Optimizer,
    MetaOptimizer,
    ProximalOptimizer,
)
from scnn.private.models.model import Model
from scnn.private.metrics import (
    update_metrics,
    init_metrics,
    format_recent_metrics,
)
from scnn.private.methods.termination_criteria import TerminationCriterion


class DoubleLoopProcedure(IterativeOptimizationProcedure):

    """An iterative optimization procedure."""

    optimizer: Optimizer

    def __init__(
        self,
        inner_optimizer: Optimizer,
        outer_optimizer: MetaOptimizer,
        inner_max_iters: int,
        outer_max_iters: int,
        inner_term_criterion: TerminationCriterion,
        outer_term_criterion: TerminationCriterion,
        name: str = "",
        divergence_check: bool = False,
        batch_size: Optional[int] = None,
        log_freq: int = ITER_LOG_FREQ,
        metric_freq: int = METRIC_FREQ,
        pre_process: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        max_total_iters: Optional[int] = None,
    ):
        """
        :param inner_optimizer: an optimizer instance that will be used to execute the inner
            optimization loop.
        :param outer_optimizer: an optimizer instance that will be used to execute the outer
            optimization loop.
        :param inner_max_iters: the maximum number of iterations to run inner optimizer per
            outer step.
        :param outer_max_iters: the maximum number of iterations to run the outer optimizer.
        :param inner_term_criterion: the criterion to use when checking for early termination
            of the inner optimization routine.
        :param outer_term_criterion: the criterion to use when checking for early termination
            of the outer optimization routine.
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
        :param max_total_iters: the maximum number of inner steps that are permitted.
        """
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.inner_max_iters = inner_max_iters
        self.outer_max_iters = outer_max_iters
        self.inner_term_criterion = inner_term_criterion
        self.outer_term_criterion = outer_term_criterion

        self.name = name
        self.divergence_check = divergence_check
        self.batch_size = batch_size

        self.log_freq = log_freq
        self.metric_freq = metric_freq

        self.pre_process = pre_process
        self.post_process = post_process
        self.max_total_iters = max_total_iters

    def reset(self):
        """Reset the optimization procedure (including any attached optimizers)
        to their original state."""
        self.outer_optimizer.reset()
        self.inner_optimizer.reset()

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
        (model, metrics_log, exit_status, objective, grad,) = self._pre_optimization(
            logger,
            model,
            initializer,
            train_data,
            test_data,
            metrics,
            final_metrics,
            self.inner_optimizer.step_size,
        )
        (X, y), f0, start_time = train_data, objective, None
        # training loop
        verbose = root.level <= INFO

        # whether or not to terminate the outer optimization procedure.
        terminate = False
        total_itrs = 0

        # outer loop
        for outer_itr in tqdm(
            range(self.outer_max_iters),
            desc="Outer " + self.name,
            disable=(not verbose),
        ):

            # inner loop says to terminate optimization.
            if terminate:
                break

            # capture time-cost of outer optimizer.

            start_time = self._get_start_time(start_time)
            # outer update
            (
                model,
                self.inner_term_criterion,
                self.inner_optimizer,
                sp_exit_state,
            ) = self.outer_optimizer.step(
                model, self.inner_term_criterion, self.inner_optimizer, X, y
            )

            # compute objective and gradient after outer update.
            objective = model.objective(X, y, batch_size=self.batch_size)
            grad = model.grad(
                X,
                y,
                batch_size=self.batch_size,
                step_size=self.inner_optimizer.step_size,
            )

            # inner loop
            for inner_itr in tqdm(
                range(self.inner_max_iters),
                desc="Inner " + self.name,
                disable=(not verbose),
            ):

                if (
                    self.max_total_iters is not None
                    and total_itrs >= self.max_total_iters
                ):
                    terminate = True
                    break
                total_itrs += 1

                start_time = self._get_start_time(start_time)

                if inner_itr % self.log_freq == 0 and verbose:
                    tqdm.write(format_recent_metrics(metrics_log, metrics))

                # outer termination criterion
                if self.outer_term_criterion(model, X, y, objective, grad):
                    exit_status["success"] = True
                    logger.info(
                        f"*Outer* termination criterion satisfied at iteration {outer_itr}/{self.outer_max_iters}. Exiting *outer* optimization loop."
                    )
                    terminate = True
                    break

                # inner termination criterion
                model = self.outer_optimizer.inner(model)
                if self.inner_term_criterion(model, X, y, objective, grad):
                    exit_status["success"] = True
                    logger.info(
                        f"*Inner* termination criterion satisfied at iteration {inner_itr}/{self.inner_max_iters}. Exiting *inner* optimization loop."
                    )
                    break

                # check for divergence
                if self.divergence_check and objective > 1000 * f0:
                    exit_status["success"] = False
                    logger.warning(
                        f"Method diverged at {inner_itr}/{self.inner_max_iters}. Exiting optimization loop."
                    )
                    break

                model, objective, sp_exit_state = self.inner_optimizer.step(
                    model, X, y, objective, grad, self.batch_size
                )

                if callback is not None:
                    model = callback(model, X, y)

                model = self.outer_optimizer.outer(model)

                # compute objective and gradient for next iteration.
                if objective is None:
                    objective = model.objective(X, y, batch_size=self.batch_size)

                grad = model.grad(
                    X,
                    y,
                    batch_size=self.batch_size,
                    step_size=self.inner_optimizer.step_size,
                )

                # calculate time and other metrics.
                if inner_itr % self.metric_freq == 0:
                    metrics_log, start_time = self._record_time(metrics_log, start_time)
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

            if inner_itr == self.inner_max_iters - 1:
                logger.warning(
                    "Max iterations reached before *inner* termination criterion was satisfied. Exiting *inner* optimization loop."
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
            self.inner_optimizer.step_size,
        )
        # collect final run-information
        if outer_itr == self.outer_max_iters - 1 or total_itrs == self.max_total_iters:
            exit_status["success"] = False
            logger.warning(
                "Max iterations reached before *outer* termination criterion was satisfied. Exiting *outer* optimization loop."
            )
        exit_status["outer_iterations"] = outer_itr + 1
        exit_status["total_inner_iterations"] = total_itrs

        return exit_status, model, metrics_log
