# src/truck_parking/rl/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from truck_parking.rl.env import Env
from truck_parking.rl.tfmr import Tfmr


class InfoTensorboardCallback(BaseCallback):
    """Log env scalars to TensorBoard."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._n_arrivals = 0
        self._n_collisions = 0
        self._n_timeouts = 0

    @staticmethod
    def _is_scalar(v: Any) -> bool:
        if isinstance(v, (int, float, np.number)):
            return True
        if isinstance(v, th.Tensor) and v.numel() == 1:
            return True
        return False

    @staticmethod
    def _to_float(v: Any) -> float:
        if isinstance(v, th.Tensor):
            return float(v.detach().cpu().item())
        return float(v)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        info0 = infos[0]
        for k, v in info0.items():
            if self._is_scalar(v):
                self.logger.record(f"env/{k}", self._to_float(v))

        if bool(info0.get("arrival", False)):
            self._n_arrivals += 1
        if bool(info0.get("collision", False)):
            self._n_collisions += 1
        if bool(info0.get("timeout", False)):
            self._n_timeouts += 1

        self.logger.record("env_counts/arrival", float(self._n_arrivals))
        self.logger.record("env_counts/collision", float(self._n_collisions))
        self.logger.record("env_counts/timeout", float(self._n_timeouts))

        return True


class Trainer:
    """PPO trainer for truck_parking."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._cfg = cfg

    def train(self) -> PPO:
        """Trains the PPO model.

        Returns:
            The trained PPO model.
        """
        art_dir = Path(self._cfg["artifact_dir"])
        chk_dir = art_dir / "checkpoints"
        art_dir.mkdir(parents=True, exist_ok=True)
        chk_dir.mkdir(parents=True, exist_ok=True)

        seed = self._cfg["seed"]
        self._set_torch_seed(seed)
        env = self._make_vec_env(seed)

        ppo = self._cfg["ppo"]
        mdl = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=self._policy_kwargs(),
            learning_rate=ppo["learning_rate"],
            n_steps=ppo["n_steps"],
            batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"],
            gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"],
            clip_range=ppo["clip_range"],
            ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"],
            tensorboard_log=str(art_dir),
            device="cuda",
            seed=seed,
            verbose=1,
        )

        ckpt_cb = CheckpointCallback(
            save_freq=self._cfg["ncheckpoint_steps"],
            save_path=str(chk_dir),
            name_prefix="chkpt",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        mdl.learn(
            total_timesteps=self._cfg["ntotal_steps"],
            callback=CallbackList([ckpt_cb, InfoTensorboardCallback()]),
            tb_log_name="ppo",
            progress_bar=True,
        )

        mdl.save(str(art_dir / "model_last.zip"))
        return mdl

    @staticmethod
    def _set_torch_seed(seed: int) -> None:
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    def _make_vec_env(self, seed: int) -> VecMonitor:
        def make_one() -> Env:
            env = Env(self._cfg["env"])
            env.reset(seed=seed)
            return env

        return VecMonitor(DummyVecEnv([make_one]))

    def _policy_kwargs(self) -> Dict[str, Any]:
        return dict(
            features_extractor_class=Tfmr,
            features_extractor_kwargs=dict(cfg=self._cfg["tfmr"]),
            net_arch=dict(
                pi=self._cfg["policy"]["pi_layers"],
                vf=self._cfg["policy"]["vf_layers"],
            ),
        )
