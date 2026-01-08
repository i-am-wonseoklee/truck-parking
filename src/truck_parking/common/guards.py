# truck_parking/common/guards.py
from __future__ import annotations

import numpy as np


def _chk_finite(x: float) -> bool:
    return bool(np.isfinite(x))


def req_finite(name: str, value: float) -> float:
    v = float(value)
    if not _chk_finite(v):
        raise ValueError(f"{name} must be a finite number. Got: {value}")
    return v


def req_positive(name: str, value: float) -> float:
    v = req_finite(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0. Got: {v}")
    return v


def req_negative(name: str, value: float) -> float:
    v = req_finite(name, value)
    if v >= 0.0:
        raise ValueError(f"{name} must be < 0. Got: {v}")
    return v


def req_non_negative(name: str, value: float) -> float:
    v = req_finite(name, value)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0. Got: {v}")
    return v


def req_bounds_ordered(
    name_min: str, v_min: float, name_max: str, v_max: float
) -> None:
    mn = req_finite(name_min, v_min)
    mx = req_finite(name_max, v_max)
    if mn > mx:
        raise ValueError(f"Invalid bounds: {name_min}({mn}) > {name_max}({mx})")


def req_in_range(name: str, value: float, low: float, high: float) -> float:
    v = req_finite(name, value)
    lo = req_finite(f"{name}.low", low)
    hi = req_finite(f"{name}.high", high)
    if lo > hi:
        raise ValueError(f"{name} range invalid: low({lo}) > high({hi})")
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}]. Got: {v}")
    return v


def req_symmetric(name_min: str, v_min: float, name_max: str, v_max: float) -> None:
    mn = req_finite(name_min, v_min)
    mx = req_finite(name_max, v_max)
    if not np.isclose(-mn, mx):
        raise ValueError(
            f"{name_min} and {name_max} must be symmetric. Got: {mn}, {mx}"
        )
