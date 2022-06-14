"""Model initializers."""
from logging import Logger
from typing import Tuple, Dict, Any, Callable

import numpy as np

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers.l2 import L2Regularizer
from scnn.private.models import ConvexMLP, ReLUMLP
from scnn.activations import sample_gate_vectors, compute_activation_patterns

# constants

# use a random initialization.
RANDOM = "random"
# initialize at 0.
ZERO = "zero"
# gate vectors
GATES = "gates"
# solve the unregularized regression problem using an iterative solver.
LEAST_SQRS = "least_squares"

# default solver config
DEFAULT_LSMR_SOLVER = {"name": "lsmr"}


def get_initializer(
    logger: Logger,
    rng: np.random.Generator,
    train_data: Tuple[lab.Tensor, lab.Tensor],
    config: Dict[str, Any],
) -> Callable[[Model], Model]:
    """Construct and return a closure which can be used to initialize a model
    before optimization.

    :param logger: a logger instance.
    :param rng: a seeded random number generator.
    :param train_data: the training set. This is required for "clever" initializations based on
        solving the un-regularized least-squares problem.
    :param config: the configuration dictionary for the initialization method.
    """

    def initialize_model(model_to_init: Model) -> Model:
        """Initialize the given model according to the given initializer
        config.

        :param model_to_init: the model to initialize.
        :returns: the initialized model.
        """
        name = config.get("name", None)
        lam = 0.0

        if model_to_init.regularizer is not None and hasattr(
            model_to_init.regularizer, "lam"
        ):
            lam = model_to_init.regularizer.lam

        if name is None:
            return model_to_init
        elif name == RANDOM:
            model_to_init.set_weights(
                lab.tensor(
                    rng.standard_normal(model_to_init.weights.shape),
                    dtype=lab.get_dtype(),
                )
            )
        elif name == GATES:
            if hasattr(model_to_init, "U"):
                model_to_init.set_weights(model_to_init.U)
            else:
                act_config = config.get("sign_patterns", {"seed": 778})

                G = sample_gate_vectors(
                    act_config["seed"], train_data[0].shape[1], act_config["n_samples"]
                )
                D, G = compute_activation_patterns(lab.to_np(train_data[0]), G)
                G = lab.tensor(G.T, dtype=lab.get_dtype())

                if isinstance(model_to_init, ReLUMLP):
                    p = G.shape[0]
                    model_to_init.p = p
                    G = lab.concatenate([lab.ravel(G), lab.ones(p)])

                model_to_init.set_weights(G)

        elif name == ZERO:
            if isinstance(model_to_init, ReLUMLP):
                raise ValueError(
                    "A (Gated) ReLU MLPs should never be initialized at zero!"
                )

            model_to_init.set_weights(lab.zeros_like(model_to_init.weights))

        elif name == LEAST_SQRS:
            # TODO: Fixme!
            raise NotImplementedError("TODO! Fix least squares initialization.")

            if not isinstance(model_to_init, ConvexMLP):
                raise ValueError(
                    "Initializing with the ridge-regression solution is only for ConvexMLPs and subclasses."
                )

            cur_backend = lab.backend
            X_np, y_np = lab.all_to_np(train_data)
            y_np = np.squeeze(y_np)
            if len(y_np.shape) > 1:
                raise ValueError(
                    "Cannot use least-squares initialization with multi-target problems!"
                )

            np_train_data = (X_np, y_np)

            np_D = lab.to_np(model_to_init.D)
            np_U = lab.to_np(model_to_init.U)

            # least squares only supports NumPy
            lab.set_backend(lab.NUMPY)
            solver_config = config.get("solver", DEFAULT_LSMR_SOLVER)

            # scale lambda to account for scaling the objective by 1/n.
            scaled_lam = lam * np_train_data[0].shape[0]
            l2 = L2Regularizer(scaled_lam)
            model_to_fit = ConvexMLP(
                model_to_init.d,
                np_D,
                np_U,
                model_to_init.kernel,
                regularizer=l2,
            )

            # solver = get_method(logger, rng, model_to_fit, np_train_data, solver_config)
            solver = None
            exit_status, fit_model, metrics_log = solver(
                logger,
                model_to_fit,
                lambda x: x,
                np_train_data,
                np_train_data,
                ([], [], []),
            )

            # reset the linear algebra backend.
            lab.set_backend(cur_backend)
            fit_model_weights = lab.tensor(fit_model.weights, dtype=lab.get_dtype())

            model_to_init.set_weights(fit_model_weights)

        else:
            raise ValueError(
                f"Initializer {name} not recognized! Please register it in 'models/initializers.py'."
            )

        return model_to_init

    return initialize_model
