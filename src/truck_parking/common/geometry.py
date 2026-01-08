# src/truck_parking/common/geometry.py
from __future__ import annotations

from typing import Tuple

import numpy as np


def wrap(angle: float) -> float:
    """Wraps angle to [-pi, pi].

    Args:
        angle (float): Angle in radians.

    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def tf_point(
    x: float, y: float, ref_x: float, ref_y: float, ref_psi: float
) -> Tuple[float, float]:
    """Transforms a 2D point to a reference frame.

    Args:
        x (float): X coordinate of the point.
        y (float): Y coordinate of the point.
        ref_x (float): X coordinate of the reference frame.
        ref_y (float): Y coordinate of the reference frame.
        ref_psi (float): Orientation of the reference frame in radians.

    Returns:
        Tuple[float, float]: Transformed point coordinates (x', y').
    """
    dx = x - ref_x
    dy = y - ref_y

    c = float(np.cos(ref_psi))
    s = float(np.sin(ref_psi))

    x_prime = dx * c + dy * s
    y_prime = -dx * s + dy * c

    return x_prime, y_prime


def tf_pose(
    x: float, y: float, psi: float, ref_x: float, ref_y: float, ref_psi: float
) -> Tuple[float, float, float]:
    """Transforms a 2D pose to a reference frame.

    Args:
        x (float): X coordinate of the pose.
        y (float): Y coordinate of the pose.
        psi (float): Orientation of the pose in radians.
        ref_x (float): X coordinate of the reference frame.
        ref_y (float): Y coordinate of the reference frame.
        ref_psi (float): Orientation of the reference frame in radians.

    Returns:
        Tuple[float, float, float]: Transformed pose (x', y', psi').
    """

    x_prime, y_prime = tf_point(x, y, ref_x, ref_y, ref_psi)
    psi_prime = wrap(psi - ref_psi)
    return x_prime, y_prime, psi_prime
