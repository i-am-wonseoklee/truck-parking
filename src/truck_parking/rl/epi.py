# src/truck-parking/rl/epi.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np

from truck_parking.core.obstacle import Obstacle
from truck_parking.core.spline import Spline
from truck_parking.core.truck import FState, TruckCfg
from truck_parking.rl.db import DB, Record


@dataclass
class Epi:
    """A container for episode data."""

    t: float = 0.0  # Elapsed time.
    s: int = 0  # Currenst arc length.
    truck_cfg: TruckCfg = None  # Truck configuration.
    obstacles: List[Obstacle] = None  # List of obstacles.
    spline: Spline = None  # Reference spline.
    state: FState = None  # Full state.


class EpiFetcher:
    """Helper class to fetch an episode multiple times."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._db = DB(path=Path(cfg["dbpath"]))
        self._nrepeats = cfg["nrepeats"]
        self._cnt = 0
        self._record: Record | None = None

    def fetch(self, prog: float) -> Epi:
        """Fetches an episode based on the given progress value.

        Args:
            prog (float): Progress value in the range [0.0, 1.0].

        Returns:
            Epi: The fetched episode.
        """
        if self._cnt == 0:
            self._record = self._db.fetch(loop=True)
            if self._record is None:
                raise RuntimeError("No records found in the database.")

        self._cnt += 1
        if self._cnt >= self._nrepeats:
            self._cnt = 0

        prog = np.clip(prog, 0.0 + 1e-6, 1.0 - 1e-6)
        s = prog * self._record.spline.length
        kstate = self._record.spline.calc_kstate(s)
        fstate = FState(
            x=kstate.x,
            y=kstate.y,
            psi=kstate.psi,
            phi=kstate.phi,
            v=0.0,
            delta=0.0,
            a=0.0,
            w=0.0,
        )

        return Epi(
            t=0.0,
            s=s,
            truck_cfg=self._record.truck_cfg,
            obstacles=self._record.obstacles,
            spline=self._record.spline,
            state=fstate,
        )
