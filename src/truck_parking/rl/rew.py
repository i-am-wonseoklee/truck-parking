# src/truck-parking/rl/rew.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping

import numpy as np

from truck_parking.common.geometry import wrap
from truck_parking.core.truck import FState
from truck_parking.rl.epi import Epi


class Rew:
    """Reward computation for truck parking environment."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._cfg = cfg

    def calc(self, epi: Epi, dt: float, state: FState) -> Dict[str, Any]:
        """Calculates the reward based on the current episode and next state.

        Args:
            epi (Epi): The current episode.
            dt (float): Time step duration.
            state (FState): The next state after taking an action for dt.

        Returns:
            Dict[str, Any]: A dictionary containing the reward and additional info.
        """
        rew_tracking = self._calc_tracking(epi, dt, state)
        rew_smoothness = self._calc_smoothness(epi, dt, state)
        rew_constraints = self._calc_constrains(epi, state)
        rew_terminal = self._calc_terminal(epi, state)

        rew = (
            rew_tracking["rew"]
            + rew_smoothness["rew"]
            + rew_constraints["rew"]
            + rew_terminal["rew"]
        )

        terminated = rew_terminal["arrival"] or rew_terminal["collision"]
        truncated = not terminated and rew_terminal["timeout"]

        return {
            "total": rew,
            "tracking": rew_tracking["rew"],
            "smoothness": rew_smoothness["rew"],
            "constraints": rew_constraints["rew"],
            "terminal": rew_terminal["rew"],
            "terminated": terminated,
            "truncated": truncated,
            "d_l": rew_tracking["d_l"],
            "e_x": rew_tracking["e_x"],
            "e_y": rew_tracking["e_y"],
            "e_psi": rew_tracking["e_psi"],
            "e_phi": rew_tracking["e_phi"],
            "jerk": rew_smoothness["jerk"],
            "o_a": rew_constraints["o_a"],
            "o_w": rew_constraints["o_w"],
            "o_phi": rew_constraints["o_phi"],
            "timeout": rew_terminal["timeout"],
            "collision": rew_terminal["collision"],
            "arrival": rew_terminal["arrival"],
            "phi": abs(state.phi),
            "v": abs(state.v),
            "a": abs(state.a),
            "delta": abs(state.delta),
            "w": abs(state.w),
        }

    def _calc_tracking(self, epi: Epi, dt: float, state: FState) -> Dict[str, float]:
        c = self._cfg["tracking"]

        v_max = min(
            epi.truck_cfg.v_max,
            abs(epi.state.v) + epi.truck_cfg.a_max * dt,
        )
        d_max = dt * v_max
        d_l = max(0.0, min(epi.spline.length - epi.s, d_max))

        ref = epi.spline.calc_kstate(epi.s + d_l)

        e_x = abs(state.x - ref.x)
        e_y = abs(state.y - ref.y)
        e_psi = abs(wrap(state.psi - ref.psi))
        e_phi = abs(wrap(state.phi - ref.phi))

        t_x = c["lambda_x"] * (e_x / c["sigma_x"]) ** 2
        t_y = c["lambda_y"] * (e_y / c["sigma_y"]) ** 2
        t_psi = c["lambda_psi"] * (e_psi / c["sigma_psi"]) ** 2
        t_phi = c["lambda_phi"] * (e_phi / c["sigma_phi"]) ** 2

        rew = -c["weight"] * np.tanh(t_x + t_y + t_psi + t_phi)

        return {
            "rew": rew,
            "d_l": d_l,
            "e_x": e_x,
            "e_y": e_y,
            "e_psi": e_psi,
            "e_phi": e_phi,
        }

    def _calc_smoothness(self, epi: Epi, dt: float, state: FState) -> Dict[str, float]:
        c = self._cfg["smoothness"]
        jerk = abs((state.a - epi.state.a) / dt)
        t_jerk = c["lambda_jerk"] * (jerk / c["sigma_jerk"]) ** 2
        rew = -c["weight"] * np.tanh(t_jerk)
        return {
            "rew": rew,
            "jerk": jerk,
        }

    def _calc_constrains(self, epi: Epi, state: FState) -> Dict[str, float]:
        c = self._cfg["constraints"]

        def over(val: float, limit: float) -> float:
            return max(0.0, abs(val) - limit)

        o_a = abs(over(state.a, epi.truck_cfg.a_max))
        o_w = abs(over(state.w, epi.truck_cfg.w_max))
        o_phi = abs(over(state.phi, epi.truck_cfg.phi_max))

        t_a = c["lambda_a"] * (o_a / c["sigma_a"]) ** 2
        t_w = c["lambda_w"] * (o_w / c["sigma_w"]) ** 2
        t_phi = c["lambda_phi"] * (o_phi / c["sigma_phi"]) ** 2

        rew = -c["weight"] * np.tanh(t_a + t_w + t_phi)

        return {
            "rew": rew,
            "o_a": o_a,
            "o_w": o_w,
            "o_phi": o_phi,
        }

    def _calc_terminal(self, epi: Epi, state: FState) -> Dict[str, float]:
        c = self._cfg["terminal"]

        timeout = False
        collision = False
        arrival = False

        if epi.t >= c["timeout"]:
            timeout = True

        trc_poly = state.kstate.calc_trc_poly(epi.truck_cfg)
        trl_poly = state.kstate.calc_trl_poly(epi.truck_cfg)
        for obst in epi.obstacles:
            if obst.chk_overlap(trc_poly) or obst.chk_overlap(trl_poly):
                collision = True
                break

        goal = epi.spline.calc_kstate(epi.spline.length)
        e_x = abs(state.x - goal.x)
        e_y = abs(state.y - goal.y)
        e_psi = abs(wrap(state.psi - goal.psi))
        e_phi = abs(wrap(state.phi - goal.phi))
        e_v = abs(state.v)
        if (
            e_x <= c["x_tol"]
            and e_y <= c["y_tol"]
            and e_psi <= c["psi_tol"]
            and e_phi <= c["phi_tol"]
            and e_v <= c["v_tol"]
        ):
            arrival = True

        rew = 0.0
        if timeout:
            rew = -c["c_timeout"]
        elif collision:
            rew = -c["c_collision"]
        elif arrival:
            rew = c["c_arrival"]

        return {
            "rew": rew,
            "timeout": timeout,
            "collision": collision,
            "arrival": arrival,
        }
