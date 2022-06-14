"""Regularizers for training neural networks by convex reformulation."""


class Regularizer:
    """Base class for all regularizers."""

    def __init__(self, lam: float):
        """Initialize the squared-error loss function."""
        self.lam = lam

    def __str__(self):
        return f"regularizer_{self.lam}"


class NeuronGL1(Regularizer):
    """A neuron-wise group-L1 regularizer.

    This regularizer produces neuron sparsity in the final model,
    meaning that some neurons will be completely inactive after training.
    The regularizer has the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^p \\|U_i\\|_2,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """

    def __str__(self):
        return f"neuron_gl1_{self.lam}"


class FeatureGL1(Regularizer):
    """A feature-wise group-L1 regularizer.

    This regularizer produces feature sparsity in the final model, meaning
    that some features will not be used after training.
    The regularizer has the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^d \\|U_{\\cdot, i}\\|_2,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """

    def __str__(self):
        return f"feature_gl1_{self.lam}"


class L2(Regularizer):
    """The standard squared-L2 norm regularizer, sometimes called weight-decay.

    The regularizer has the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^p \\|U_{i}\\|^2_2,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """

    def __str__(self):
        return f"l2_{self.lam}"


class L1(Regularizer):
    """The L1 norm regularizer.

    The regularizer has the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^p \\|U_{i}\\|_1,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """

    def __str__(self):
        return f"l1_{self.lam}"
