"""Tools for training models with feature sparsity.


TODO:
    - Extract support from linear model.
    - Extract support by forward/backward.
"""
from typing import List, Tuple, Optional

import numpy as np
import lab

from scnn.regularizers import L1, FeatureGL1, Regularizer
from scnn.models import ConvexReLU, ConvexGatedReLU, LinearModel, Model
from scnn.solvers import RFISTA, Optimizer
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors
from scnn.optimize import optimize_path, optimize_model

from scnn.private.utils.data.transforms import train_test_split


class SupportFinder:
    """Base class for support finders."""

    def __init__(self, valid_prop: float = 0.2):
        """ """
        self.valid_prop = valid_prop

    def __call__(
        self,
        model: Model,
        regularizer: Optional[Regularizer],
        solver: Optimizer,
        train_set: Tuple[np.ndarray, np.ndarray],
        device,
        dtype,
        seed,
    ) -> List[int]:
        raise NotImplementedError(
            "A SupportFinder must be callable to obtain \
                a support vector."
        )


class LinearSupportFinder(SupportFinder):
    """Find support by training a penalized linear model."""

    def __init__(self, lambda_path: List[float], valid_prop: float = 0.2):
        """ """
        self.lambda_path = lambda_path
        self.valid_prop = valid_prop

    def __call__(
        self,
        model: Model,
        regularizer: Optional[Regularizer],
        solver: Optimizer,
        train_set: Tuple[np.ndarray, np.ndarray],
        device,
        dtype,
        seed,
    ) -> List[int]:
        """ """
        path: List[Regularizer]
        X, y = train_set
        n, d = X.shape
        _, c = y.shape

        train, valid = train_test_split(
            train_set[0],
            train_set[1],
            valid_prop=self.valid_prop,
            split_seed=seed,
        )

        if c == 1:
            path = [L1(lam) for lam in self.lambda_path]
        else:
            path = [FeatureGL1(lam) for lam in self.lambda_path]

        model = LinearModel(d, c)
        solver = RFISTA(model)
        metrics = Metrics(train_mse=True, test_mse=True)

        path_models, path_metrics = optimize_path(
            model,
            solver,
            path,
            metrics,
            train[0],
            train[1],
            valid[0],
            valid[1],
            device=device,
            dtype=dtype,
            seed=seed,
        )

        # find best validation accuracy and extract support.
        best_mse = None
        best_index = None

        for i, m in enumerate(path_metrics):
            if best_mse is None or m.test_mse[-1] < best_mse:
                best_mse = m.test_mse[-1]
                best_index = i

        non_zeros = (
            lab.sum(lab.abs(path_models[best_index].parameters[0]), axis=0)
            != 0
        )

        support = lab.arange(d)[non_zeros].tolist()

        if len(support) == 0:
            # failure mode
            return lab.arange(d).tolist()
        else:
            return support


class ForwardBackward(SupportFinder):
    """Support finder based on the forward/backward method."""

    def __init__(self, forward: bool = True, valid_prop: float = 0.2):
        """ """

        self.valid_prop = valid_prop
        self.forward = forward

    def __call__(
        self,
        model: Model,
        regularizer: Optional[Regularizer],
        solver: Optimizer,
        train_set: Tuple[np.ndarray, np.ndarray],
        device,
        dtype,
        seed,
    ) -> List[int]:

        if not isinstance(model, (ConvexReLU, ConvexGatedReLU)):
            raise ValueError(
                "Forward-backward only supports convex reformulations."
            )

        p = model.p

        X, y = train_set
        n, d = X.shape
        _, c = y.shape

        train, valid = train_test_split(
            train_set[0],
            train_set[1],
            valid_prop=self.valid_prop,
            split_seed=seed,
        )

        remaining_features = np.arange(d).tolist()

        # find best validation accuracy and extract support.
        if self.forward:
            active_features = []
        else:
            active_features = remaining_features

        best_support = None
        best_crit = None

        for i in range(d):
            next_best_crit: float = None
            next_features: List[int] = None

            for feature in remaining_features:

                if self.forward:
                    features = active_features + [feature]
                else:
                    features = active_features.copy()
                    features.remove(feature)

                # Add or remove feature from dataset
                f_train = train[0][:, features], train[1]
                f_valid = valid[0][:, features], valid[1]

                # Update model (ie. resample gates).
                G = sample_gate_vectors(seed, len(features), p)
                f_model = model.__class__(G, model.c)

                # hard code metrics for now.
                metrics = Metrics(train_mse=True, test_mse=True)
                # train model
                f_model, f_metrics = optimize_model(
                    f_model,
                    solver,
                    metrics,
                    f_train[0],
                    f_train[1],
                    f_valid[0],
                    f_valid[1],
                    regularizer=regularizer,
                    return_convex=True,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                )

                crit = f_metrics.test_mse[-1]

                # check criterion
                if next_best_crit is None or crit <= next_best_crit:
                    best_feature = feature
                    next_features = features
                    next_best_crit = crit

                if best_crit is None or crit <= best_crit:
                    best_support = features
                    best_crit = crit

            active_features = next_features
            remaining_features.remove(best_feature)

        return best_support
