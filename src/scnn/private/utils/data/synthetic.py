"""Functions for generating synthetic datasets.

Public Functions:

    gen_classification_data: generate a synthetic classification dataset with
        targets given by a random neural network.

    gen_regression_data: generate a synthetic regression dataset with targets
        given by a noisy linear model.

    gen_sparse_regression_problem: generate synthetic data from a variety of
        simple feature-sparse planted models.
"""

from typing import Tuple, Literal, Optional, Union, Callable
import math

import numpy as np
from scipy.stats import ortho_group  # type: ignore

Transform = Literal["cosine", "polynomial"]

# local types

Dataset = Tuple[np.ndarray, np.ndarray]


def gen_classification_data(
    data_seed: int,
    n: int,
    n_test: int,
    d: int,
    hidden_units: int = 50,
    kappa: float = 1.0,
) -> Tuple[Dataset, Dataset]:
    """Create a binary classification dataset with a random Gaussian design
    matrix and targets given by a two-layer neural network with random Gaussian
    weights.

    If `kappa` is supplied, then the design matrix satisfies
    :math:`\\kappa(X) \\approx \\text{kappa}`.

    Args:
        data_seed: the seed to use when generating the synthetic dataset.
        n: number of examples in dataset.
        n_test: number of test examples.
        d: number of features for each example.
        hidden_units: (optional) the number of hidden units in the neural
            network.
        kappa: (optional) the (approximate) condition number of the train/test
            design matrices.

    Returns:
        A training set `(X_train, y_train)` and test set `(X_test, y_test)`.
    """
    rng = np.random.default_rng(seed=data_seed)

    w = rng.random(d * hidden_units + hidden_units)
    w1 = w[: d * hidden_units].reshape(hidden_units, d)
    w2 = w[d * hidden_units :].reshape(1, hidden_units)

    Sigma = sample_covariance_matrix(rng, d, kappa)

    X = []
    y = []
    n_pos, n_neg = 0, 0
    n_total = n + n_test

    # simple rejection sampling
    while n_pos + n_neg < n_total:
        xi = rng.multivariate_normal(np.zeros(d), cov=Sigma)
        # compute forward pass
        yi = np.maximum(xi @ w1.T, 0) @ w2.T
        if yi <= 0 and n_neg < math.ceil(n_total):
            y.append(-1)
            X.append(xi)
            n_neg += 1
        elif yi > 0 and n_pos < math.ceil(n_total):
            y.append(1)
            X.append(xi)
            n_pos += 1

    X_np = np.array(X)
    y_np = np.array(y).reshape(-1, 1)

    # shuffle dataset.
    indices = np.arange(n_total)
    rng.shuffle(indices)
    X_np, y_np = X_np[indices], y_np[indices]

    train_set = (X_np[:n], y_np[:n])
    test_set = (X_np[n:], y_np[n:])

    return train_set, test_set


def gen_regression_data(
    data_seed: int,
    n: int,
    n_test: int,
    d: int,
    c: int = 1,
    sigma: float = 0,
    kappa: float = 1,
) -> Tuple[Dataset, Dataset, np.ndarray]:
    """Create a regression dataset with a random Gaussian design matrix.

    If `kappa` is supplied, then the design matrix satisfies
    :math:`\\kappa(X) \\approx \\text{kappa}`.

    Args:
        data_seed: the seed to use when generating the synthetic dataset.
        n: number of examples in dataset.
        n_test: number of test examples.
        d: number of features for each example.
        c: dimension of the targets.
            Defaults to scalar targets (`c = 1`).
        sigma: variance of (Gaussian) noise added to targets.
            Defaults to `0` for a noiseless model.
        kappa: condition number of :math:`\\mathbb{E_X}[X^\\top X]`.
            Defaults to 1.

    Returns:
        A training set `(X_train, y_train)`, test set `(X_test, y_test)`, and
        the group-truth model `w_opt`.
    """
    rng = np.random.default_rng(seed=data_seed)
    # sample "true" model
    w_opt = rng.standard_normal((d, c))
    Sigma = sample_covariance_matrix(rng, d, kappa)

    X = rng.multivariate_normal(np.zeros(d), cov=Sigma, size=n + n_test)
    y = np.dot(X, w_opt)

    if sigma != 0:
        y = y + rng.normal(0, scale=sigma)

    train_set = (X[:n], y[:n])
    test_set = (X[n:], y[n:])

    return train_set, test_set, w_opt


def gen_sparse_regression_problem(
    data_seed: int,
    n: int,
    n_test: int,
    d: int,
    sigma: float = 0,
    kappa: float = 1,
    nnz: Optional[int] = None,
    transform: Optional[Union[Transform, Callable]] = None,
) -> Tuple[Dataset, Dataset, np.ndarray]:
    """Sample data from a feature-sparse planted model.

    Create a realizable regression problem by sampling data from a
    simple planted model. A variety of planted models are available; see
    the `transform` argument. Inspired by code form Tolga Ergen.

    Args:
        data_seed: the seed to use when generating the synthetic dataset.
        n: number of examples in dataset.
        n_test: number of test examples.
        d: number of features for each example.
        sigma: variance of (Gaussian) noise added to targets.
            Defaults to `0` for a noiseless model.
        kappa: condition number of E[X.T X].
            Defaults to 1.
        nnz: number of non-zeros zeros in the true model.
        transform: a non-linear transformation. This must be `None`,
            `'cosine'`, `'polynomial'`, `'product'`, or a callable function that applies a
            custom transformation.

    Returns:
        A training set `(X_train, y_train)`, test set `(X_test, y_test)`, and
        the group-truth model `w_opt`.
    """

    rng = np.random.default_rng(seed=data_seed)
    # sample "true" model
    w_opt = rng.standard_normal((d))

    # true model is sparse
    non_zero_indices = np.arange(d)
    if nnz is not None:
        non_zero_indices = rng.choice(d, size=nnz, replace=False)
        mask = np.zeros_like(w_opt)
        mask[non_zero_indices] = 1.0
        w_opt = w_opt * mask

    # sample covariance matrix.
    Sigma = sample_covariance_matrix(rng, d, kappa)

    X = rng.multivariate_normal(np.zeros(d), cov=Sigma, size=n + n_test)
    y = np.dot(X, w_opt)

    if transform is None:
        # no transform
        y = y
    elif transform == "cosine":
        # cosine transform
        y = np.cos(y)
    elif transform == "cubic":
        # simple cubic
        y = y + (y ** 2) / 2 + (y ** 3) / 6
    elif transform == "product":
        y = np.sign(np.prod(X[:, non_zero_indices], axis=1))
    else:
        try:
            y = transform(y)
        except TypeError:
            raise ValueError(
                f"Transform {transform} not recognized. It must be a \
                        predefined transform, a callable function, or `None`."
            )

    # add noise
    if sigma != 0:
        y = y + rng.normal(0, scale=sigma)

    train_set = (X[:n], y[:n])
    test_set = (X[n:], y[n:])

    return train_set, test_set, w_opt


def sample_covariance_matrix(
    rng: np.random.Generator, d: int, kappa: float
) -> np.ndarray:
    """Sample a covariance matrix with a specific condition number.

    This functions samples a symmetric positive-definite matrix
    with condition number exactly `kappa`. The minimum eigenvalue is `1` and
    the maximum is `kappa`, while the remaining eigenvalues are distributed
    uniformly at random in the interval `[1, kappa]`.

    Args:
        rng: a NumPy random number generator.
        d: the dimensionality of the covariance matrix.
        kappa: condition number of the covariance matrix.

    Returns:
        A (n,d) matrix :math:`\\Sigma` with condition number
            `kappa`.
    """

    # sample random orthonormal matrix
    Q = ortho_group.rvs(d, random_state=rng)
    # sample eigenvalues so that lambda_1 / lambda_d = kappa.
    eigs = rng.uniform(low=1, high=kappa, size=d - 2)
    eigs = np.concatenate([np.array([kappa, 1]), eigs])
    # compute covariance
    Sigma = np.dot(Q.T, np.multiply(np.expand_dims(eigs, axis=1), Q))

    return Sigma
