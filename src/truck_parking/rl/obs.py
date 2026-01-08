# src/truck-parking/rl/obs.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import gymnasium as gym
import numpy as np

from truck_parking.common.geometry import tf_point, tf_pose
from truck_parking.core.obstacle import Obstacle
from truck_parking.core.spline import Spline
from truck_parking.core.truck import FState, TruckCfg
from truck_parking.rl.epi import Epi


class Obs:
    """Helper class to build observations."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._cfg = cfg

    def bld_spec(self) -> gym.spaces.Dict:
        """Build observation space specification.

        Returns:
            Observation space specification.
        """
        return gym.spaces.Dict(
            {
                "param_loncon": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "param_latcon": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "param_geom": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(9,),
                    dtype=np.float32,
                ),
                "param_art": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "param_act": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
                "obsts": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._cfg["n_obsts"], 4, 2),
                    dtype=np.float32,
                ),
                "obsts_mask": gym.spaces.MultiBinary(self._cfg["n_obsts"]),
                "refs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._cfg["n_refs"], 4),
                    dtype=np.float32,
                ),
                "dir": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    def bld_obs(self, epi: Epi) -> Dict[str, np.ndarray]:
        """Builds an observation.

        Args:
            episode: An episode data container.

        Returns:
            An observation.
        """
        return {
            "param_loncon": self._bld_obs_param_loncon(epi.truck_cfg),
            "param_latcon": self._bld_obs_param_latcon(epi.truck_cfg),
            "param_geom": self._bld_obs_param_geom(epi.truck_cfg),
            "param_art": self._bld_obs_param_art(epi.truck_cfg),
            "param_act": self._bld_obs_param_act(epi.truck_cfg),
            "state": self._bld_obs_state(epi.state),
            "obsts": self._bld_obs_obsts(epi.obstacles, epi.state),
            "obsts_mask": self._bld_obs_obsts_mask(epi.obstacles),
            "refs": self._bld_obs_refs(
                epi.s,
                epi.truck_cfg,
                epi.spline,
                epi.state,
            ),
            "dir": np.array([1 if epi.spline.dir > 0 else -1], dtype=np.float32),
        }

    def _bld_obs_param_loncon(self, tconf: TruckCfg) -> np.ndarray:
        return np.array(
            [
                tconf.v_min,
                tconf.v_max,
                tconf.a_min,
                tconf.a_max,
            ],
            dtype=np.float32,
        )

    def _bld_obs_param_latcon(self, tconf: TruckCfg) -> np.ndarray:
        return np.array(
            [
                tconf.delta_min,
                tconf.delta_max,
                tconf.w_min,
                tconf.w_max,
            ],
            dtype=np.float32,
        )

    def _bld_obs_param_geom(self, tconf: TruckCfg) -> np.ndarray:
        return np.array(
            [
                tconf.L_trc,
                tconf.L_trl,
                tconf.L_f_trc,
                tconf.L_f_trl,
                tconf.L_r_trc,
                tconf.L_r_trl,
                tconf.W_trc,
                tconf.W_trl,
                tconf.rho,
            ],
            dtype=np.float32,
        )

    def _bld_obs_param_art(self, tconf: TruckCfg) -> np.ndarray:
        return np.array(
            [
                tconf.phi_min,
                tconf.phi_max,
            ],
            dtype=np.float32,
        )

    def _bld_obs_param_act(self, tconf: TruckCfg) -> np.ndarray:
        return np.array(
            [
                tconf.tau_v,
                tconf.tau_delta,
            ],
            dtype=np.float32,
        )

    def _bld_obs_state(self, state: FState) -> np.ndarray:
        return np.array(
            [
                state.phi,
                state.v,
                state.delta,
                state.a,
                state.w,
            ],
            dtype=np.float32,
        )

    def _bld_obs_obsts(self, obsts: List[Obstacle], state: FState) -> np.ndarray:
        obsts_arr = np.zeros((self._cfg["n_obsts"], 4, 2), dtype=np.float32)
        for i, obst in enumerate(obsts):
            if i >= self._cfg["n_obsts"]:
                break
            for j, (x, y) in enumerate(obst.coords[:-1]):
                x_pr, y_pr = tf_point(x, y, state.x, state.y, state.psi)
                obsts_arr[i, j, 0] = x_pr
                obsts_arr[i, j, 1] = y_pr
        return obsts_arr

    def _bld_obs_obsts_mask(self, obsts: List[Obstacle]) -> np.ndarray:
        mask = np.zeros((self._cfg["n_obsts"],), dtype=np.int8)
        for i in range(min(len(obsts), self._cfg["n_obsts"])):
            mask[i] = 1
        return mask

    def _bld_obs_refs(
        self, s: float, tcfg: TruckCfg, spline: Spline, state: FState
    ) -> np.ndarray:
        n = self._cfg["n_refs"]
        t = self._cfg["t_refs"]
        v_max = tcfg.v_max
        a_max = tcfg.a_max
        v = abs(state.v)

        t_acc = min(t, max(0, v_max - v) / a_max)
        d_max = v * t_acc + 0.5 * a_max * t_acc**2 + v_max * (t - t_acc)

        s_begin = s
        s_end = min(s_begin + d_max, spline.length)
        ss = np.linspace(s_begin, s_end, n)

        refs = np.zeros((n, 4), dtype=np.float32)
        for j, s in enumerate(ss):
            kstate = spline.calc_kstate(s)
            x, y, psi = tf_pose(
                kstate.x,
                kstate.y,
                kstate.psi,
                state.x,
                state.y,
                state.psi,
            )
            refs[j, 0] = x
            refs[j, 1] = y
            refs[j, 2] = psi
            refs[j, 3] = kstate.phi

        return refs
