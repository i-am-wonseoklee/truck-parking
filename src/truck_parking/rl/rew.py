# src/truck-parking/rl/rew.py
from __future__ import annotations

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
        rew_progress = self._calc_progress(epi, dt, state)
        rew_tracking = self._calc_tracking(epi, dt, state)
        rew_smoothness = self._calc_smoothness(epi, dt, state)
        rew_constraints = self._calc_constraints(epi, state)
        rew_terminal = self._calc_terminal(epi, state)

        rew = (
            rew_progress["rew"]
            + rew_tracking["rew"]
            + rew_smoothness["rew"]
            + rew_constraints["rew"]
            + rew_terminal["rew"]
        )

        terminated = rew_terminal["arrival"] or rew_terminal["collision"]
        truncated = not terminated and rew_terminal["timeout"]

        return {
            "total": rew,
            "terminated": terminated,
            "truncated": truncated,
            "rew_progress": rew_progress["rew"],
            "rew_tracking": rew_tracking["rew"],
            "rew_smoothness": rew_smoothness["rew"],
            "rew_constraints": rew_constraints["rew"],
            "rew_terminal": rew_terminal["rew"],
            "progress": rew_progress["progress"],
            "d_l": rew_tracking["d_l"],
            "e_pos": rew_tracking["e_pos"],
            "e_psi": rew_tracking["e_psi"],
            "e_phi": rew_tracking["e_phi"],
            "da": rew_smoothness["da"],
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

    def _calc_progress(self, epi: Epi, dt: float, state: FState) -> Dict[str, float]:
        c = self._cfg["progress"]

        d_max = dt * epi.truck_cfg.v_max * 2  # Conservative estimate.
        s_prev = epi.s
        s_curr = epi.spline.calc_s(state.x, state.y, srange=(s_prev, s_prev + d_max))
        progress = s_curr - s_prev

        rew = c["weight"] * np.tanh(progress / c["sigma_progress"])

        return {
            "rew": rew,
            "progress": progress,
        }

    def _calc_tracking(self, epi: Epi, dt: float, state: FState) -> Dict[str, float]:
        c = self._cfg["tracking"]

        d_max = dt * epi.truck_cfg.v_max
        d_l = max(0.0, min(epi.spline.length - epi.s, d_max))

        ref = epi.spline.calc_kstate(epi.s + d_l)

        e_pos = np.hypot(state.x - ref.x, state.y - ref.y)
        e_psi = abs(wrap(state.psi - ref.psi))
        e_phi = abs(wrap(state.phi - ref.phi))

        t_pos = np.tanh(c["lambda_pos"] * (e_pos / c["sigma_pos"]) ** 2)
        t_psi = np.tanh(c["lambda_psi"] * (e_psi / c["sigma_psi"]) ** 2)
        t_phi = np.tanh(c["lambda_phi"] * (e_phi / c["sigma_phi"]) ** 2)

        rew = -c["weight"] * (t_pos + t_psi + t_phi) / 3.0

        return {
            "rew": rew,
            "d_l": d_l,
            "e_pos": e_pos,
            "e_psi": e_psi,
            "e_phi": e_phi,
        }

    def _calc_smoothness(self, epi: Epi, dt: float, state: FState) -> Dict[str, float]:
        c = self._cfg["smoothness"]
        da = abs(state.a - epi.state.a)
        t_da = np.tanh((da / c["sigma_da"]) ** 2)
        rew = -c["weight"] * t_da
        return {
            "rew": rew,
            "da": da,
        }

    def _calc_constraints(self, epi: Epi, state: FState) -> Dict[str, float]:
        c = self._cfg["constraints"]

        def over(val: float, limit: float) -> float:
            return max(0.0, abs(val) - limit)

        o_a = abs(over(state.a, epi.truck_cfg.a_max))
        o_w = abs(over(state.w, epi.truck_cfg.w_max))
        o_phi = abs(over(state.phi, epi.truck_cfg.phi_max))

        t_a = np.tanh(c["lambda_a"] * (o_a / c["sigma_a"]) ** 2)
        t_w = np.tanh(c["lambda_w"] * (o_w / c["sigma_w"]) ** 2)
        t_phi = np.tanh(c["lambda_phi"] * (o_phi / c["sigma_phi"]) ** 2)

        rew = -c["weight"] * (t_a + t_w + t_phi) / 3.0

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
        e_pos = np.hypot(state.x - goal.x, state.y - goal.y)
        e_psi = abs(wrap(state.psi - goal.psi))
        e_phi = abs(wrap(state.phi - goal.phi))
        e_v = abs(state.v)
        if (
            e_pos <= c["pos_tol"]
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
