"""Optimization procedure for PyTorch models."""
from logging import Logger, root, INFO
from typing import Dict, Any, List, Tuple, Callable, Optional
import timeit

from tqdm.auto import tqdm  # type: ignore
import torch
import numpy as np

import lab

from scnn.private.methods.optimization_procedures.optimization_procedure import (
    METRIC_FREQ,
    EPOCH_LOG_FREQ,
)
from scnn.private.methods.optimization_procedures.iterative import (
    IterativeOptimizationProcedure,
)

from scnn.private.methods.optimizers import Optimizer
from scnn.private.models.model import Model
from scnn.private.metrics import update_metrics, init_metrics, format_recent_metrics
from scnn.private.methods.termination_criteria import TerminationCriterion
import scnn.private.loss_functions as loss_fns


class TorchLoop(IterativeOptimizationProcedure):

    """Optimization loop for PyTorch optimizers and models."""

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        batch_size: int,
        term_criterion: TerminationCriterion,
        name: str = "",
        divergence_check: bool = True,
        log_freq: int = EPOCH_LOG_FREQ,
        metric_freq: int = METRIC_FREQ,
        scheduler: Optional[Callable] = None,
        pre_process: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        """
        :param optimizer: an optimizer instance implementing the 'step' method.
        :param max_epochs: the maximum number of epochs to run the optimizer.
        :param batch_size: the batch_size to use.
        :param term_criterion: the criterion to use when checking for early termination.
        :param name: an optional name to display when printing the progress bar.
        :param divergence_check: an whether or not the optimization procedure should
            check for divergence behavior and terminate early.
        :param log_freq: the frequency at which to information during optimization.
        :param metric_freq: the frequency (in epochs) at which to collect metrics, like objective,
            gradient-norm, etc, during optimization.
        :param scheduler: a PyTorch-style step-size scheduler.
        :param pre_process: (optional) a function to call on the model after the it is
            initialized but *before* optimization starts.
        :param post_process: (optional) a function to call on the model *after* optimization
            is complete.
        """
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.term_criterion = term_criterion

        self.name = name
        self.divergence_check = divergence_check

        self.log_freq = log_freq
        self.metric_freq = metric_freq
        self.scheduler = scheduler

        self.pre_process = pre_process
        self.post_process = post_process

    def reset(self):
        """Reset the optimization procedure (including any attached optimizers)
        to their original state."""
        # TODO: reset the optimizer.

    def _get_optimizer_step_size(self):
        return self.optimizer.state_dict()["param_groups"][0]["lr"]

    def __call__(
        self,
        logger: Logger,
        model: Model,
        initializer: Callable[[Model], Model],
        train_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor],
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
        # enable autodiff engine
        lab.toggle_autodiff(True)

        # pre-optimization setup
        with torch.no_grad():
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
                self._get_optimizer_step_size(),
            )
            (X, y), f0, start_time = train_data, objective, None

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)

        tensor_dataset = torch.utils.data.TensorDataset(X, y)
        training_loader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=lab.torch_rng,
        )

        # training loop
        verbose = root.level <= INFO
        for epoch in tqdm(
            range(self.max_epochs), desc=self.name, disable=(not verbose)
        ):
            start_time = self._get_start_time(start_time)

            if epoch % self.log_freq == 0 and verbose:
                tqdm.write(format_recent_metrics(metrics_log, metrics))

            # check termination criteria
            with torch.no_grad():
                if self.term_criterion(model, X, y, objective, grad):
                    exit_status["success"] = True
                    logger.info(
                        f"Termination criterion satisfied at iteration {epoch}/{self.max_epochs}. Exiting optimization loop."
                    )
                    break

                # check for divergence
                if (
                    self.divergence_check
                    and objective > 1000 * f0
                    or torch.isnan(objective)
                ):
                    exit_status["success"] = False
                    logger.warning(
                        f"Method diverged at {epoch}/{self.max_epochs}. Exiting optimization loop."
                    )
                    break

            for itr, (xi, yi) in enumerate(training_loader):

                self.optimizer.zero_grad()
                # hard-code squared error at the moment.
                obj = model.objective(xi, yi)
                obj.backward()
                self.optimizer.step()

            if callback is not None:
                model = callback(model, X, y)

            # call the scheduler once per epoch in the PyTorch style.
            if self.scheduler is not None:
                self.scheduler.step()

            # update metrics every epoch.
            if epoch % self.metric_freq == 0:
                with torch.no_grad():

                    # calculate time for one iteration.
                    metrics_log, start_time = self._record_time(metrics_log, start_time)
                    objective = model.objective(X, y)
                    grad = model.grad(X, y)
                    metrics_log = update_metrics(
                        metrics_log,
                        model,
                        {"step_size": self._get_optimizer_step_size()},
                        train_data,
                        test_data,
                        metrics,
                        objective,
                        grad,
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
            self._get_optimizer_step_size(),
        )

        # collect run-information
        if epoch == self.max_epochs - 1:
            exit_status["success"] = False
            logger.warning(
                "Max iterations reached before termination criterion was satisfied. Exiting optimization loop."
            )
        exit_status["epochs"] = epoch + 1

        # disable autodiff engine
        lab.toggle_autodiff(False)

        return exit_status, model, metrics_log
