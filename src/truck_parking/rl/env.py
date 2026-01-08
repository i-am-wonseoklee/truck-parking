# src/truck-parking/rl/env.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Optional

import gymnasium as gym
import numpy as np

from truck_parking.rl.epi import Epi, EpiFetcher
from truck_parking.rl.obs import Obs
from truck_parking.rl.rew import Rew


class Env(gym.Env):
    """Gymnasium environment for truck parking."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self._cfg = cfg
        self._rng = np.random.default_rng()
        self._fetcher = EpiFetcher(self._cfg["episode_fetcher"])
        self._obs = Obs(self._cfg["obs"])
        self._rew = Rew(self._cfg["rew"])
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = self._obs.bld_spec()
        self._epi = Epi()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment to start a new episode.

        Args:
            seed (Optional[int]): Seed for the random number generator.
            options (Optional[Dict[str, Any]]): Additional options for resetting the environment.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._epi = self._fetcher.fetch(prog=self._rng.uniform(0.0, 1.0))
        return self._obs.bld_obs(self._epi), {}  # No reset info.

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Advances the environment by one time step.

        Args:
            action (np.ndarray): Action array containing desired velocity and steering angle.

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]: Observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        # Aliases for readability.
        dt = self._cfg["dt"]
        tcfg = self._epi.truck_cfg
        spline = self._epi.spline

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise ValueError(f"Invalid action shape, {action.shape}")

        v = action[0] * tcfg.v_max * (1 if spline.dir > 0 else -1)
        delta = action[1] * tcfg.delta_max

        next_state = self._epi.state.step(tcfg, dt, v, delta)
        rewinfo = self._rew.calc(self._epi, dt, next_state)

        d_max = dt * tcfg.v_max * 2  # Conservative estimate.
        self._epi.t += dt
        self._epi.s = spline.calc_s(
            next_state.x,
            next_state.y,
            (
                self._epi.s,
                self._epi.s + d_max,
            ),
        )
        self._epi.state = next_state

        return (
            self._obs.bld_obs(self._epi),
            rewinfo["total"],
            rewinfo["terminated"],
            rewinfo["truncated"],
            rewinfo,
        )
