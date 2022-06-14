"""Loss functions for training neural networks by convex reformulation.

Notes:
    - We only support the squared loss for the time being.
"""


class SquaredLoss:
    """The squared-error loss function.

    Given predictions :math:`f(X)` and targets :math:`y`, this loss function
    has the form,

    .. math:: L(f(X), y) = \\frac{1}{2} \\|f(X) - y\\|_2^2.
    """

    def __init__(self):
        """ """
        pass
