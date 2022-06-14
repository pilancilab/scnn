"""Core subroutines for augmented Lagrangian methods and ADMM."""
from typing import Tuple

import numpy as np

import lab


def update_multipliers(
    e_multipliers: lab.Tensor,
    i_multipliers: lab.Tensor,
    e_gap: lab.Tensor,
    i_gap: lab.Tensor,
    delta: float,
) -> Tuple[lab.Tensor, lab.Tensor]:
    """
    :param e_multipliers: an estimate of the optimal dual parameters for the equality constraints.
    :param i_multipliers: an estimate of the optimal dual parameters for the inequality constraints.
    :param e_gap: the violation of the equality constraint.
    :param i_gap: the violation of the inequality constraint.
    :param delta: the strength of the quadratic penalty in the augmented Lagrangian.
    :returns: the updated dual parameters.
    """

    return e_multipliers + delta * e_gap, lab.smax(i_multipliers + delta * i_gap, 0)


def acc_update_multipliers(
    e_multipliers: lab.Tensor,
    i_multipliers: lab.Tensor,
    e_gap: lab.Tensor,
    i_gap: lab.Tensor,
    delta: float,
    e_v: lab.Tensor,
    i_v: lab.Tensor,
    t: float,
) -> Tuple[lab.Tensor, lab.Tensor, lab.Tensor, lab.Tensor, float]:
    """
    :param e_multipliers: an estimate of the optimal dual parameters for the equality constraints.
    :param i_multipliers: an estimate of the optimal dual parameters for the inequality constraints.
    :param e_gap: the violation of the equality constraint.
    :param i_gap: the violation of the inequality constraint.
    :param delta: the strength of the quadratic penalty in the augmented Lagrangian.
    :returns: the updated dual parameters.
    """

    t_plus = 1 + np.sqrt(1 + 4 * t ** 2) / 2
    beta = (t - 1) / t_plus

    e_plus, i_plus = e_v + delta * e_gap, lab.smax(i_v + delta * i_gap, 0)

    e_v_plus = e_plus + beta * (e_plus - e_multipliers)
    i_v_plus = i_plus + beta * (i_plus - i_multipliers)

    return e_plus, i_plus, e_v_plus, i_v_plus, t_plus
