# src/truck_parking/core/truck.py
from __future__ import annotations

from dataclasses import Field, dataclass, field, fields
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from truck_parking.common.geometry import wrap
from truck_parking.common.guards import (
    req_finite,
    req_negative,
    req_positive,
    req_bounds_ordered,
    req_in_range,
    req_symmetric,
)

EPS: float = 1e-6
HALF_PI: float = 0.5 * np.pi


@dataclass(slots=True, frozen=True)
class TruckCfg:
    """Configuration for the truck consists of a tractor and a trailer."""

    v_min: float = -5.0  # Max reverse speed < 0 (m/s).
    v_max: float = 5.0  # Max forward speed > 0 (m/s).
    a_min: float = -2.0  # Max deceleration < 0 (m/s^2).
    a_max: float = 2.0  # Max acceleration > 0 (m/s^2).
    delta_min: float = -0.5  # Max right steering angle < 0 (rad).
    delta_max: float = 0.5  # Max left steering angle > 0 (rad).
    w_min: float = -0.3  # Max right steering rate < 0 (rad/s).
    w_max: float = 0.3  # Max left steering rate > 0 (rad/s).

    # The `trc` and `trl` suffixes stand for `tractor` and `trailer`, respectively.
    L_trc: float = 4.0  # Wheelbase of the tractor (m).
    L_trl: float = 8.0  # Wheelbase of the trailer (m).
    L_f_trc: float = 1.0  # Front overhang of the tractor (m).
    L_f_trl: float = 1.0  # Front overhang of the trailer (m).
    L_r_trc: float = 2.0  # Rear overhang of the tractor (m).
    L_r_trl: float = 1.0  # Rear overhang of the trailer (m).
    W_trc: float = 3.0  # Width of the tractor (m).
    W_trl: float = 4.0  # Width of the trailer (m).

    rho: float = -1.0  # Dist. from tractor rear axle center to articulation point (m).

    # The articulation angle is defined as phi = psi_tractor - psi_trailer.
    phi_min: float = -HALF_PI * 0.8  # Min articulation angle (rad).
    phi_max: float = HALF_PI * 0.8  # Max articulation angle (rad).

    tau_v: float = 0.263  # Time constant for speed actuator dynamics (s).
    tau_delta: float = 0.1  # Time constant for steering actuator dynamics (s).

    _poly_trc: Polygon = field(
        init=False, default_factory=Polygon, repr=False, compare=False
    )  # Polygon of the tractor body at origin aligned with x-axis.
    _poly_trl: Polygon = field(
        init=False, default_factory=Polygon, repr=False, compare=False
    )  # Polygon of the trailer body at origin aligned with x-axis.

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_poly_trc",
            box(
                minx=-self.L_r_trc,
                miny=-0.5 * self.W_trc,
                maxx=self.L_f_trc + self.L_trc,
                maxy=0.5 * self.W_trc,
            ),
        )
        object.__setattr__(
            self,
            "_poly_trl",
            box(
                minx=-self.L_r_trl,
                miny=-0.5 * self.W_trl,
                maxx=self.L_f_trl + self.L_trl,
                maxy=0.5 * self.W_trl,
            ),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "TruckCfg":
        """Creates a TruckCfg instance from a dictionary.

        Args:
            d (Mapping[str, float]): Configuration dictionary.

        Returns:
            TruckCfg: Truck configuration instance.
        """
        obj = cls(**{f.name: d[f.name] for f in cls._gen_init_fields()})
        obj._validate()
        return obj

    def to_dict(self) -> Dict[str, float]:
        """Converts the TruckCfg instance to a dictionary.

        Returns:
            Dict[str, float]: Configuration dictionary.
        """
        return {f.name: float(getattr(self, f.name)) for f in self._gen_init_fields()}

    @property
    def poly_trc(self) -> Polygon:
        """Tractor polygon aligned with x-axis at origin.

        Returns:
            Polygon: Tractor polygon.
        """
        return self._poly_trc

    @property
    def poly_trl(self) -> Polygon:
        """Trailer polygon aligned with x-axis at origin.

        Returns:
            Polygon: Trailer polygon.
        """
        return self._poly_trl

    @classmethod
    def _gen_init_fields(cls) -> Iterator[Field[Any]]:
        return (f for f in fields(cls) if f.init)

    def _validate(self) -> None:
        for f in self._gen_init_fields():
            req_finite(f.name, float(getattr(self, f.name)))

        req_negative("v_min", self.v_min)
        req_positive("v_max", self.v_max)
        req_symmetric("v_min", self.v_min, "v_max", self.v_max)
        req_negative("a_min", self.a_min)
        req_positive("a_max", self.a_max)
        req_symmetric("a_min", self.a_min, "a_max", self.a_max)
        req_negative("delta_min", self.delta_min)
        req_positive("delta_max", self.delta_max)
        req_symmetric("delta_min", self.delta_min, "delta_max", self.delta_max)
        req_negative("w_min", self.w_min)
        req_positive("w_max", self.w_max)
        req_symmetric("w_min", self.w_min, "w_max", self.w_max)
        req_negative("phi_min", self.phi_min)
        req_positive("phi_max", self.phi_max)
        req_symmetric("phi_min", self.phi_min, "phi_max", self.phi_max)
        req_bounds_ordered("v_min", self.v_min, "v_max", self.v_max)
        req_bounds_ordered("a_min", self.a_min, "a_max", self.a_max)
        req_bounds_ordered("delta_min", self.delta_min, "delta_max", self.delta_max)
        req_bounds_ordered("w_min", self.w_min, "w_max", self.w_max)
        req_bounds_ordered("phi_min", self.phi_min, "phi_max", self.phi_max)
        req_in_range("delta_min", self.delta_min, low=-HALF_PI, high=HALF_PI)
        req_in_range("delta_max", self.delta_max, low=-HALF_PI, high=HALF_PI)
        req_positive("L_trc", self.L_trc)
        req_positive("L_trl", self.L_trl)
        req_positive("L_f_trc", self.L_f_trc)
        req_positive("L_f_trl", self.L_f_trl)
        req_positive("L_r_trc", self.L_r_trc)
        req_positive("L_r_trl", self.L_r_trl)
        req_positive("W_trc", self.W_trc)
        req_positive("W_trl", self.W_trl)
        req_positive("tau_v", self.tau_v)
        req_positive("tau_delta", self.tau_delta)


@dataclass(slots=True, frozen=True)
class KState:
    """Kinematic state of the truck."""

    x: float = 0.0  # Tractor rear axle center x (m).
    y: float = 0.0  # Tractor rear axle center y (m).
    psi: float = 0.0  # Tractor heading angle (rad).
    phi: float = 0.0  # Articulation angle (rad).

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", req_finite("x", self.x))
        object.__setattr__(self, "y", req_finite("y", self.y))
        object.__setattr__(self, "psi", wrap(req_finite("psi", self.psi)))
        object.__setattr__(self, "phi", wrap(req_finite("phi", self.phi)))

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "KState":
        """Creates a KState instance from a dictionary.

        Args:
            d (Mapping[str, float]): State dictionary.

        Returns:
            KState: Kinematic state instance.
        """
        return KState(
            x=d["x"],
            y=d["y"],
            psi=d["psi"],
            phi=d["phi"],
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts the KState instance to a dictionary.

        Returns:
            Dict[str, float]: State dictionary.
        """
        return {
            "x": float(self.x),
            "y": float(self.y),
            "psi": float(self.psi),
            "phi": float(self.phi),
        }

    def step(self, cfg: TruckCfg, dt: float, v: float, delta: float) -> "KState":
        """Applies commanded (v, delta) for dt seconds and returns the next state.

        Notes:
        - This method assumes perfect tracking of (v, delta) without actuator dynamics.
        - The resulting state is computed using a simple Euler integration step.

        Args:
            cfg (TruckCfg): Truck configuration.
            dt (float): Time step duration (s).
            v (float): Commanded speed (m/s).
            delta (float): Commanded steering angle (rad).

        Returns:
            KState: Next kinematic state.
        """
        dt = req_positive("dt", dt)
        v = req_in_range("v", v, cfg.v_min, cfg.v_max)
        delta = req_in_range("delta", delta, cfg.delta_min, cfg.delta_max)

        psi = self.psi
        phi = self.phi

        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        psi_dot = v * np.tan(delta) / cfg.L_trc
        phi_dot = psi_dot - (
            (cfg.rho * psi_dot * np.cos(phi) + v * np.sin(phi)) / cfg.L_trl
        )

        return KState(
            x=self.x + x_dot * dt,
            y=self.y + y_dot * dt,
            psi=wrap(psi + psi_dot * dt),
            phi=wrap(phi + phi_dot * dt),
        )

    def calc_trl_pose(self, cfg: TruckCfg) -> Tuple[float, float, float]:
        """Calculates the trailer pose (x, y, psi) from the current kinematic state.

        Args:
            cfg (TruckCfg): Truck configuration.

        Returns:
            Tuple[float, float, float]: Trailer pose (x, y, psi).
        """
        art_x = self.x + cfg.rho * np.cos(self.psi)
        art_y = self.y + cfg.rho * np.sin(self.psi)

        trl_psi = self.psi - self.phi
        trl_x = art_x - cfg.L_trl * np.cos(trl_psi)
        trl_y = art_y - cfg.L_trl * np.sin(trl_psi)

        return trl_x, trl_y, trl_psi

    def calc_trc_poly(self, cfg: TruckCfg) -> Polygon:
        """Calculates the tractor polygon at the current kinematic state.

        Args:
            cfg (TruckCfg): Truck configuration.

        Returns:
            Polygon: Tractor polygon in the world frame.
        """
        poly = rotate(cfg.poly_trc, self.psi, origin=(0, 0), use_radians=True)
        poly = translate(poly, self.x, self.y)
        return poly

    def calc_trl_poly(self, cfg: TruckCfg) -> Polygon:
        """Calculates the trailer polygon at the current kinematic state.

        Args:
            cfg (TruckCfg): Truck configuration.

        Returns:
            Polygon: Trailer polygon in the world frame.
        """
        trl_x, trl_y, trl_psi = self.calc_trl_pose(cfg)
        poly = rotate(cfg.poly_trl, trl_psi, origin=(0, 0), use_radians=True)
        poly = translate(poly, trl_x, trl_y)
        return poly


@dataclass(slots=True, frozen=True)
class FState:
    """Full state of the truck."""

    x: float = 0.0  # Tractor rear axle center x (m).
    y: float = 0.0  # Tractor rear axle center y (m).
    psi: float = 0.0  # Tractor heading angle (rad).
    phi: float = 0.0  # Articulation angle (rad).
    v: float = 0.0  # Speed (m/s).
    delta: float = 0.0  # Steering angle (rad).
    a: float = 0.0  # Acceleration (m/s^2).
    w: float = 0.0  # Steering rate (rad/s).

    _kstate: KState = field(
        init=False, default_factory=KState, repr=False, compare=False
    )  # Cached kinematic state.

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", req_finite("x", self.x))
        object.__setattr__(self, "y", req_finite("y", self.y))
        object.__setattr__(self, "psi", wrap(req_finite("psi", self.psi)))
        object.__setattr__(self, "phi", wrap(req_finite("phi", self.phi)))
        object.__setattr__(self, "v", req_finite("v", self.v))
        object.__setattr__(
            self,
            "delta",
            req_in_range("delta", self.delta, -HALF_PI, HALF_PI),
        )
        object.__setattr__(self, "a", req_finite("a", self.a))
        object.__setattr__(self, "w", req_finite("w", self.w))
        object.__setattr__(
            self,
            "_kstate",
            KState(x=self.x, y=self.y, psi=self.psi, phi=self.phi),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "FState":
        """Creates a FState instance from a dictionary.

        Args:
            d (Mapping[str, float]): State dictionary.

        Returns:
            FState: Full state instance.
        """
        return FState(
            x=d["x"],
            y=d["y"],
            psi=d["psi"],
            phi=d["phi"],
            v=d["v"],
            delta=d["delta"],
            a=d["a"],
            w=d["w"],
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts the FState instance to a dictionary.

        Returns:
            Dict[str, float]: State dictionary.
        """
        return {
            "x": float(self.x),
            "y": float(self.y),
            "psi": float(self.psi),
            "phi": float(self.phi),
            "v": float(self.v),
            "delta": float(self.delta),
            "a": float(self.a),
            "w": float(self.w),
        }

    def step(self, cfg: TruckCfg, dt: float, v: float, delta: float) -> "FState":
        """Applies commanded (v, delta) for dt seconds and returns the next state.

        Notes:
        - This method models first-order actuator dynamics for (v, delta) tracking.
        - The resulting state is computed using a semi-implicit Euler integration step.

        Args:
            cfg (TruckCfg): Truck configuration.
            dt (float): Time step duration (s).
            v (float): Commanded speed (m/s).
            delta (float): Commanded steering angle (rad).

        Returns:
            FState: Next full state.
        """
        dt = req_positive("dt", dt)
        v = req_in_range("v", v, cfg.v_min, cfg.v_max)
        delta = req_in_range("delta", delta, cfg.delta_min, cfg.delta_max)

        cur_v = self.v
        cur_delta = self.delta

        tau_v = cfg.tau_v
        tau_delta = cfg.tau_delta

        new_v = cur_v + (1 - np.exp(-dt / tau_v)) * (v - cur_v)
        new_delta = cur_delta + (1 - np.exp(-dt / tau_delta)) * (delta - cur_delta)
        new_a = (new_v - cur_v) / dt
        new_w = (new_delta - cur_delta) / dt

        psi = self.psi
        phi = self.phi

        x_dot = new_v * np.cos(psi)
        y_dot = new_v * np.sin(psi)
        psi_dot = new_v * np.tan(new_delta) / cfg.L_trc
        phi_dot = psi_dot - (
            (cfg.rho * psi_dot * np.cos(phi) + new_v * np.sin(phi)) / cfg.L_trl
        )

        return FState(
            x=self.x + x_dot * dt,
            y=self.y + y_dot * dt,
            psi=wrap(psi + psi_dot * dt),
            phi=wrap(phi + phi_dot * dt),
            v=new_v,
            delta=new_delta,
            a=new_a,
            w=new_w,
        )

    @property
    def kstate(self) -> KState:
        """Kinematic state corresponding to the full state.

        Returns:
            KState: Kinematic state.
        """
        return self._kstate
