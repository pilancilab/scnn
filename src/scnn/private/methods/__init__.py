"""Methods."""

# ===== module exports ===== #

from .optimization_procedures import (
    OptimizationProcedure,
    IterativeOptimizationProcedure,
    TorchLoop,
    DoubleLoopProcedure,
    METRIC_FREQ,
    ITER_LOG_FREQ,
    EPOCH_LOG_FREQ,
)
from .core import (
    ls,
    gd_ls,
    proximal_gradient_step,
    proximal_gradient_ls,
    fista_step,
    fista_ls,
    update_multipliers,
    acc_update_multipliers,
)

from .optimizers import (
    Optimizer,
    ProximalOptimizer,
    ProximalLSOptimizer,
    MetaOptimizer,
    GD,
    GDLS,
    PGD,
    PGDLS,
    FISTA,
    AugmentedLagrangian,
)

from .line_search import (
    Backtracker,
    MultiplicativeBacktracker,
    LSCondition,
    QuadraticBound,
    DiagQB,
    Armijo,
    StepSizeUpdater,
    KeepNew,
    KeepOld,
    ForwardTrack,
    Lassplore,
)

from .callbacks import (
    ObservedSignPatterns,
    ConeDecomposition,
    ProximalCleanup,
    ApproximateConeDecomposition,
)

from .external_solver import LinearSolver

from .cvxpy import (
    CVXPYSolver,
    CVXPYGatedReLUSolver,
    CVXPYReLUSolver,
    MinL2Decomposition,
    MinL1Decomposition,
    FeasibleDecomposition,
    MinRelaxedL2Decomposition,
    SOCPDecomposition,
)

from .termination_criteria import (
    GradientNorm,
    StepLength,
    ConstrainedHeuristic,
    LagrangianGradNorm,
)


__all__ = [
    "OptimizationProcedure",
    "IterativeOptimizationProcedure",
    "TorchLoop",
    "DoubleLoopProcedure",
    "ObservedSignPatterns",
    "ConeDecomposition",
    "ApproximateConeDecomposition",
    "ProximalCleanup",
    "ls",
    "gradient_step",
    "gd_ls",
    "proximal_gradient_step",
    "proximal_gradient_ls",
    "fista_step",
    "fista_ls",
    "update_multipliers",
    "acc_update_multipliers",
    "Optimizer",
    "ProximalOptimizer",
    "ProximalLSOptimizer",
    "MetaOptimizer",
    "GD",
    "GDLS",
    "PGD",
    "PGDLS",
    "FISTA",
    "AugmentedLagrangian",
    "LinearSolver",
    "CVXPYSolver",
    "CVXPYGatedReLUSolver",
    "CVXPYReLUSolver",
    "MinL2Decomposition",
    "MinL1Decomposition",
    "FeasibleDecomposition",
    "MinRelaxedL2Decomposition",
    "SOCPDecomposition",
    "Backtracker",
    "MultiplicativeBacktracker",
    "LSCondition",
    "QuadraticBound",
    "DiagQB",
    "Armijo",
    "StepSizeUpdater",
    "KeepNew",
    "KeepOld",
    "ForwardTrack",
    "Lassplore",
    "GradientNorm",
    "StepLength",
    "ConstrainedHeuristic",
    "LagrangianGradNorm",
]
