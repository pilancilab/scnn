"""Optimization methods for training neural networks by convex reformulation.

Notes:
    We only support the squared loss at the moment.

Todo:
    - Implement the cone decomposition optimizer for training ReLU model by
        (1) training a Gated ReLU model and (2) decomposing that model onto
        the conic difference.
"""
from typing import List

from .models import (
    Model,
    LinearModel,
    ConvexReLU,
    NonConvexReLU,
    ConvexGatedReLU,
    NonConvexGatedReLU,
)


class Optimizer:
    """Base class for optimizers.

    Attributes:
        model: the model that should be optimized.
    """

    def __init__(self, model: Model):
        """Initialize the optimizer.

        Args:
            model: the model to be optimized. It will be checked for
                compatibility with the optimizer.
        """
        self.model = model

    def cpu_only(self) -> bool:
        return False


class RFISTA(Optimizer):
    """Fast proximal-gradient solver for Gated ReLU models.

    This optimizer solves the convex Gated ReLU training problem by directly
    minimizing the convex reformulation,

    .. math:: F(u) = L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i}),
        y\\right) + \\lambda R(u),

    where :math:`L` is a convex loss function, :math:`R` is a regularizer,
    and :math:`\\lambda` is the regularization strength.

    Attributes:
        model: the model to be optimized.
        max_iters: the maximum number of iterations to run the solver.
        tol: the tolerance for terminating the optimization procedure early;
            `tol` will be checked against the squared norm of the minimum-norm
            subgradient.
    """

    def __init__(
        self, model: Model, max_iters: int = 10000, tol: float = 1e-6
    ):
        """
        Args:
            model: the model to be optimized. It will be checked for
                compatibility with the optimizer.
            max_iters: the maximum number of iterations to run the solver.
            tol: the tolerance for terminating the optimization procedure.
        """

        if not isinstance(model, (ConvexGatedReLU, LinearModel)):
            raise ValueError(
                "R-FISTA can only be used to train Gated ReLU models and \
                        linear model."
            )

        super().__init__(model)
        self.max_iters = max_iters
        self.tol = tol


class AL(Optimizer):
    """Augmented Lagrangian (AL) method for ReLU model.

    This optimizer solves the convex ReLU training problem by forming the
    "augmented Lagrangian",

    .. math:: \\mathcal{L}(v, w, \\gamma, \\xi) = F(v,w) + \\delta \\sum_{D_i}
        \\left[\\|(\\frac{\\gamma_i}{\\delta} - (2D_i - I)X v_i)_+\\|_2^2 +
        \\|(\\frac{\\xi_i}{\\delta} - (2D_i - I)X v_i)_+\\|_2^2 \\right],

    where :math:`\\delta > 0` is the penalty strength, :math:`(\\gamma, \\xi)`
    are the dual parameters, and

    .. math:: F(v,w) = L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X (v_{i} -
        w_{i}), y\\right) + \\lambda R(v, w),

    is the regularized training loss.

    The AL method alternates between the "primal" problem of minimizing
    :math:`\\mathcal{L}(v, w, \\gamma, \\xi)` in terms of :math:`v, w` and
    the updates to the dual parameters, :math:`\\gamma, \\xi`.
    This procedure will return an primal-dual optimal pair
    :math:`(v,w), (\\gamma, \\xi)` in the (dual) iteration limit.

    The solver will terminate early if an approximate KKT point is computed.

    Attributes:
        model: the model to be optimized.
        max_primal_iters: the maximum number of iterations to run the primal
            optimization method before exiting.
        max_dual_iters: the maximum number of dual updates that will be
            performed before exiting.
        tol: the maximum squared l2-norm of the gradient of the Lagrangian
            function for the KKT conditions to be considered approximately
            satisfied.
        constraint_tol: the maximum violation of the constraints permitted
            for the KKT conditions to be considered approximately
            satisfied.
        delta: the initial penalty strength.
    """

    def __init__(
        self,
        model: Model,
        max_primal_iters: int = 10000,
        max_dual_iters: int = 10000,
        tol: float = 1e-6,
        constraint_tol: float = 1e-6,
        delta: float = 1000,
    ):
        """
        Args:
            model: the model to be optimized. It will be checked for
                compatibility with the optimizer.
            max_primal_iters: the maximum number of iterations to run the
                primal optimization method before exiting.
            max_dual_iters: the maximum number of dual updates that will be
                performed before exiting.
            tol: the maximum squared l2-norm of the gradient of the Lagrangian
                function for the KKT conditions to be considered approximately
                satisfied.
            constraint_tol: the maximum violation of the constraints permitted
                for the KKT conditions to be considered approximately
                satisfied.
            delta: the initial penalty strength.
        """
        if not isinstance(model, (ConvexReLU)):
            raise ValueError(
                "The AL optimizer can only be used to train ReLU models."
            )

        super().__init__(model)
        self.max_primal_iters = max_primal_iters
        self.max_dual_iters = max_dual_iters
        self.tol = tol
        self.constraint_tol = constraint_tol
        self.delta = delta


class LeastSquaresSolver(Optimizer):
    """Direct solver for the unregularized or :math:`\\ell_2`-regularized
    Gated ReLU training problem.

    The Gated ReLU problem with no regularization or :math:`\\ell_2`
    regularization is

    .. math:: F(u) = \\frac{1}{2}\\|\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i}
        - y\\|_2^2 + \\lambda \sum_{D_i \\in \\mathcal{D}} \\|u_i\\|_2^2.

    This is a convex quadratic problem equivalent to ridge regression which
    this optimizer solves using conjugate-gradient (CG) type methods.
    Either LSMR [1] or LSQR [2] can be used; implementations are provided by
    `scipy.sparse.linalg
    <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_.
    See `SOL <https://web.stanford.edu/group/SOL/download.html>`_ for details
    on these methods.

    Attributes:
        model: the model that should be optimized.
        solver: the underlying CG-type solver to use.
        max_iters: the maximum number of iterations to run the optimization
            method.
        tol: the tolerance for terminating the optimization procedure early.

    Notes:
        - This solver only supports computation on CPU. The user's choice of
            backend will be overridden if necessary.

        - This solver only supports :class:`L2 <scnn.regularizers.L2>`
            regularization or no regularization.

    References:
        [1] D. C.-L. Fong and M. A. Saunders, LSMR: An iterative algorithm
            for sparse least-squares problems, SIAM J. Sci. Comput. 33:5,
            2950-2971, published electronically Oct 27, 2011.

        [2] C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse
        linear equations and sparse least squares, TOMS 8(1), 43-71 (1982)
    """

    def __init__(
        self,
        model: Model,
        solver: str = "lsmr",
        max_iters: int = 1000,
        tol=1e-6,
    ):
        """
        Args:
            model: the model to be optimized. It will be checked for
                compatibility with the optimizer.
            solver: the underlying CG-type solver to use.
                Must be one of "LSMR" or "LSQR".
            max_iters: the maximum number of iterations to run the CG-method.
            tol: the tolerance for terminating the optimization procedure
                early.
        """
        if not isinstance(model, (ConvexGatedReLU)):
            raise ValueError(
                "The least-squares solver only supports Gated ReLU models."
            )

        super().__init__(model)

        self.solver = solver
        self.max_iters = max_iters
        self.tol = tol

    def cpu_only(self) -> bool:
        return True


class CVXPYSolver(Optimizer):
    """Solve convex reformulations using `CVXPY <https://www.cvxpy.org>`_ as a
    interface to different interior-point solvers.

    `CVXPY <https://www.cvxpy.org>`_ provides a framework for denoting and
    solving convex optimization problems. The framework is compatible with a
    variety of solvers (mainly interior point methods); see
    `choosing a solver
    <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`_
    for details. Note that some solvers may require additional libraries
    and/or licences to be installed.


    Attributes:
        model: the model that should be optimized.
        solver: the underlying solver to use with CVXPY.
        solver_kwargs: a dictionary of keyword arguments to be passed directly
            to the underlying solver.
        clean_sol: whether or not to clean the solution using a
            proximal-gradient step.
     Notes:
         - This solver only supports computation on CPU. The user's choice of
            device will be overridden if necessary.
    """

    supported_solvers: List[str] = ["ecos", "cvxopt", "scs", "gurobi", "mosek"]

    def __init__(
        self, model: Model, solver, solver_kwargs={}, clean_sol=False
    ):
        """
        Args:
            model: the model to be optimized. It will be checked for
                compatibility with the optimizer.
            solver: a string identifying the solver to use with CVXPY. Only
                'ecos', 'cvxopt', 'scs', 'gurobi', and 'mosek' are supported.
                See `choosing a solver
                <https://www.cvxpy.org/tutorial/advanced/index.html#choosing
                -a-solver>`_ for details.
            solver_kwargs: keyword arguments that will be passed directly to
                the underlying solver. See `solver options
                <https://www.cvxpy.org/tutorial/advanced/index.html#setting
                -solver-options>`_ for details.
            clean_sol: whether or not to clean the solution using a
                proximal-gradient step. This is only supported for Gated ReLU
                problems.

        """
        super().__init__(model)

        if clean_sol and isinstance(model, ConvexReLU):
            raise ValueError(
                "Cleaning solutions using a proximal-gradient step is \
                only supported for unconstrained problems."
            )

        if solver not in self.supported_solvers:
            raise ValueError(
                f"CVXPYSolver does not support {solver}; it only supports \
                {self.supported_solvers} for now."
            )

        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.clean_sol = clean_sol

    def cpu_only(self) -> bool:
        return True


class ExactConeDecomposition(Optimizer):
    """Two-step method for approximately optimizing ReLU models.

    ConeDecomposition first solves the Gated ReLU problem,

    .. math:: \\min_{u} L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i}),
        y\\right) + \\lambda R(u),

    and then decomposes the solution :math:`u^*` onto the Minkowski differences
    :math:`K_i - K_i` to approximate the ReLU training problem. The cone
    decomposition is solved exactly using interior-point methods and
    CVXPY...
    """

    def __init__(self):
        raise NotImplementedError(
            "Exact cone decomposition is not \
                implemented yet."
        )


class ApproximateConeDecomposition(Optimizer):
    """Two-step method for approximately optimizing ReLU models.

    ConeDecomposition first solves the Gated ReLU problem,

    .. math:: \\min_{u} L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i}),
        y\\right) + \\lambda R(u),

    and then decomposes the solution :math:`u^*` onto the Minkowski differences
    :math:`K_i - K_i` to approximate the ReLU training problem. The cone
    decomposition is approximated by solving

    .. math:: \\min_{w} \\sum_{D_i \\in \\mathcal{D}}\\left[
        \\| \\left(\\tilde X_i w - \\min\\{\\tilde X_i u^*, 0\\}\\right)_+\\|_2^2
        + \\rho \\|w_i\\|_2^2 \\right],

    where :math:`\\tilde X_i = (2D_i - I) X`.
    The regularization :math:`\\rho` controls the quality of the approximation;
    taking :math:`\\rho \\rightarrow 0` will return a feasible solution.
    A feasible solution is guaranteed to preserve the value of the loss
    :math:`L`, but can substantially blow-up the model norm. As such, it is
    only an approximation to the ReLU training problem when
    :math:`\\lambda > 0`.

    Attributes:
        model: the model that should be optimized.
        max_iters: the maximum number of iterations to run the R-FISTA sovler.
        tol: the tolerance for terminating R-FISTA early.
        rho: the strength of the penalty term in the decomposition program.
        d_max_iters: the maximum number of iterations for the decomposition
            program.
        d_tol: the tolerance for terminating the decomposition program.
    """

    def __init__(
        self,
        model: Model,
        max_iters: int = 10000,
        tol: float = 1e-6,
        rho: float = 1e-10,
        d_max_iters: int = 10000,
        d_tol: float = 1e-10,
    ):
        """Initialize the optimizer."""
        if not isinstance(model, (ConvexGatedReLU)):
            raise ValueError(
                "Cone decomposition methods are only compatible with \
                Gated ReLU models."
            )

        super().__init__(model)
        self.rfista = RFISTA(model, max_iters, tol)
        self.rho = rho
        self.d_max_iters = d_max_iters
        self.d_tol = d_tol
