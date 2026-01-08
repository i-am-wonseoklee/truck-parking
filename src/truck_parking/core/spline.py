# src/truck_parking/core/spline.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

from truck_parking.common.geometry import wrap
from truck_parking.core.truck import KState


class Spline:
    """An arc-length parameterized spline built from a discrete state sequence."""

    def __init__(self, dir: int, states: List[KState]) -> None:
        self._dir = dir
        self._xs = np.array([ks.x for ks in states], dtype=float)
        self._ys = np.array([ks.y for ks in states], dtype=float)
        self._psis = np.unwrap(np.array([ks.psi for ks in states], dtype=float))
        self._phis = np.unwrap(np.array([ks.phi for ks in states], dtype=float))
        self._tbl = self._bld_tbl(self._xs, self._ys)
        self._ls = LineString([(ks.x, ks.y) for ks in states])

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Spline:
        """Creates a Spline instance from a dictionary.

        Args:
            d (Mapping[str, Any]): Dictionary containing spline data.

        Returns:
            Spline: Spline instance created from the dictionary.
        """
        dir = int(d["dir"])
        states = [KState.from_dict(ks) for ks in d["states"]]
        return Spline(dir=dir, states=states)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Spline instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the Spline instance.
        """
        return {
            "dir": self._dir,
            "states": [
                {
                    "x": float(self._xs[i]),
                    "y": float(self._ys[i]),
                    "psi": float(wrap(self._psis[i])),
                    "phi": float(wrap(self._phis[i])),
                }
                for i in range(len(self._xs))
            ],
        }

    def calc_s(self, x: float, y: float, srange: Tuple[float, float]) -> float:
        """Calculates the arc-length of the closest configuration on the spline within the given range.
        The resulting arc-length canniot be outside the specified range.

        Args:
            x (float): X coordinate of the point.
            y (float): Y coordinate of the point.
            srange (Tuple[float, float]): Range of arc-length to consider (srange[0] must be less than or equal to srange[1]).

        Returns:
            float: Arc-length of the closest configuration on the spline.
        """
        s_min = max(0, srange[0])
        s_max = min(self.length, srange[1])
        if s_min >= s_max:
            return s_min

        sub = substring(self._ls, s_min, s_max, normalized=False)
        s = s_min + sub.project(Point(x, y))

        return np.clip(s, s_min, s_max)

    def calc_kstate(self, s: float) -> KState:
        """Calculates the configuration on the spline at the given arc-length parameter.

        Args:
            s (float): Arc-length parameter.

        Returns:
            KState: Configuration at the given arc-length parameter.
        """
        x = np.interp(s, self._tbl, self._xs)
        y = np.interp(s, self._tbl, self._ys)
        psi = wrap(np.interp(s, self._tbl, self._psis))
        phi = wrap(np.interp(s, self._tbl, self._phis))
        return KState(x=x, y=y, psi=psi, phi=phi)

    @property
    def dir(self) -> int:
        """Returns the direction of the spline.

        Returns:
            int: Direction of the spline (1 for forward, -1 for backward).
        """
        return self._dir

    @property
    def length(self) -> float:
        """Returns the total length of the spline.

        Returns:
            float: Total length of the spline.
        """
        return float(self._tbl[-1])

    @property
    def xs(self) -> np.ndarray:
        """Returns the x coordinates of the spline.

        Returns:
            np.ndarray: X coordinates of the spline.
        """
        return self._xs

    @property
    def ys(self) -> np.ndarray:
        """Returns the y coordinates of the spline.

        Returns:
            np.ndarray: Y coordinates of the spline.
        """
        return self._ys

    @property
    def psis(self) -> np.ndarray:
        """Returns the orientations of the spline.

        Returns:
            np.ndarray: Orientations of the spline.
        """
        return self._psis

    @property
    def phis(self) -> np.ndarray:
        """Returns the steering angles of the spline.

        Returns:
            np.ndarray: Steering angles of the spline.
        """
        return self._phis

    @staticmethod
    def _bld_tbl(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        tbl = np.zeros(len(xs), dtype=float)
        for i in range(1, len(xs)):
            tbl[i] = tbl[i - 1] + float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        return tbl
