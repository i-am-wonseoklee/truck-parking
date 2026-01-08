# src/truck-parking/rl/db.py
from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from truck_parking.core.obstacle import Obstacle
from truck_parking.core.planner import Planner, PlannerCfg
from truck_parking.core.spline import Spline
from truck_parking.core.truck import KState, TruckCfg


_SQL_CREATE_SCENARIOS = """
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id INTEGER PRIMARY KEY,
    scenario_json TEXT NOT NULL
);
"""

_SQL_CREATE_TRAJECTORIES = """
CREATE TABLE IF NOT EXISTS trajectories (
    scenario_id INTEGER NOT NULL,
    trajectory_id INTEGER NOT NULL,
    truck_cfg_json TEXT NOT NULL,
    planner_cfg_json TEXT NOT NULL,
    spline_json TEXT NOT NULL,
    obstacles_json TEXT NOT NULL,
    png BLOB NOT NULL,
    PRIMARY KEY (scenario_id, trajectory_id),
    FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id) ON DELETE CASCADE
);
"""

_SQL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trajectories_scenario
ON trajectories(scenario_id);
"""

_SQL_INSERT_SCENARIO = """
INSERT OR REPLACE INTO scenarios (scenario_id, scenario_json)
VALUES (?, ?);
"""

_SQL_INSERT_TRAJECTORY = """
INSERT OR REPLACE INTO trajectories (
    scenario_id,
    trajectory_id,
    truck_cfg_json,
    planner_cfg_json,
    spline_json,
    obstacles_json,
    png
) VALUES (?, ?, ?, ?, ?, ?, ?);
"""

_SQL_SELECT_FIRST = """
SELECT
    scenario_id,
    trajectory_id,
    planner_cfg_json,
    truck_cfg_json,
    spline_json,
    obstacles_json
FROM trajectories
ORDER BY scenario_id ASC, trajectory_id ASC
LIMIT 1;
"""

_SQL_SELECT_NEXT = """
SELECT
    scenario_id,
    trajectory_id,
    planner_cfg_json,
    truck_cfg_json,
    spline_json,
    obstacles_json
FROM trajectories
WHERE (scenario_id > ?)
   OR (scenario_id = ? AND trajectory_id > ?)
ORDER BY scenario_id ASC, trajectory_id ASC
LIMIT 1;
"""


@dataclass(frozen=True, slots=True)
class Record:
    """A record to be fetched from the database."""

    scenario_id: int  # Scenario identifier.
    trajectory_id: int  # Trajectory identifier within the scenario.
    truck_cfg: TruckCfg  # Truck configuration.
    planner_cfg: PlannerCfg  # Planner configuration.
    spline: Spline  # The reference spline.
    obstacles: List[Obstacle]  # List of obstacles.


class DB:
    """SQLite-backed database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cursor: Tuple[int, int] | None = None
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def bld(self, path: Path) -> None:
        """Builds the database from a scenario directory.

        Args:
            path (Path): Path to the directory containing scenario yaml files.
        """
        if self._path.exists():
            raise FileExistsError(f"Database file already exists at {self._path}")

        yamls = sorted(path.glob("scenario_*.yaml"))
        if not yamls:
            raise ValueError(f"No scenario yaml files found in {path}")

        conn = self._connect()
        if not conn:
            raise RuntimeError("Failed to connect to the database.")

        self._init_schema(conn)
        for scenario_id, scenario_yaml in enumerate(yamls):
            scenario_cfg = yaml.safe_load(scenario_yaml.read_text())

            tcfg = TruckCfg.from_dict(scenario_cfg["truck_cfg"])
            pcfg = PlannerCfg.from_dict(scenario_cfg["planner_cfg"])
            init = KState.from_dict(scenario_cfg["init"])
            goal = KState.from_dict(scenario_cfg["goal"])
            obstacles = [
                Obstacle.from_dict(obst) for obst in scenario_cfg.get("obstacles", [])
            ]
            planner = Planner(pcfg, tcfg)

            splines = planner.plan(init, goal, obstacles)
            if not splines:
                print(f"Warning: Planning failure for scenario {scenario_yaml.name}")
                continue

            conn.execute(
                _SQL_INSERT_SCENARIO,
                (scenario_id, json.dumps(scenario_cfg)),
            )

            for trajectory_id, spline in enumerate(splines):
                conn.execute(
                    _SQL_INSERT_TRAJECTORY,
                    (
                        scenario_id,
                        trajectory_id,
                        json.dumps(tcfg.to_dict()),
                        json.dumps(pcfg.to_dict()),
                        json.dumps(spline.to_dict()),
                        json.dumps([obst.to_dict() for obst in obstacles]),
                        self._render(tcfg, pcfg, obstacles, spline),
                    ),
                )

            print(f"{scenario_yaml.name}: Inserted {len(splines)} trajectories")
            conn.commit()

        conn.close()

    def fetch(self, *, loop: bool = True) -> Record | None:
        """Gets the next record from the database.

        Args:
            loop (bool): If True, loop back to the first record when reaching the end.

        Returns:
            Record | None: The next record, or None if no more records are available.
        """
        conn = self._connect()
        try:
            if self._cursor is None:
                row = conn.execute(_SQL_SELECT_FIRST).fetchone()
            else:
                s, e = self._cursor
                row = conn.execute(_SQL_SELECT_NEXT, (s, s, e)).fetchone()

            if row is None and loop:
                self._cursor = None
                row = conn.execute(_SQL_SELECT_FIRST).fetchone()

            if row is None:
                return None

            scenario_id = row[0]
            trajectory_id = row[1]
            planner_cfg_json = row[2]
            truck_cfg_json = row[3]
            spline_json = row[4]
            obstacles_json = row[5]

            self._cursor = (int(scenario_id), int(trajectory_id))

            return Record(
                scenario_id=int(scenario_id),
                trajectory_id=int(trajectory_id),
                truck_cfg=TruckCfg.from_dict(json.loads(truck_cfg_json)),
                planner_cfg=PlannerCfg.from_dict(json.loads(planner_cfg_json)),
                spline=Spline.from_dict(json.loads(spline_json)),
                obstacles=[Obstacle.from_dict(d) for d in json.loads(obstacles_json)],
            )
        finally:
            conn.close()

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.execute(_SQL_CREATE_SCENARIOS)
        conn.execute(_SQL_CREATE_TRAJECTORIES)
        conn.execute(_SQL_CREATE_INDEX)
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _render(
        self, tcfg: TruckCfg, pcfg: PlannerCfg, obsts: List[Obstacle], spline: Spline
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(pcfg.x_min, pcfg.x_max)
        ax.set_ylim(pcfg.y_min, pcfg.y_max)
        ax.set_aspect("equal", "box")
        ax.grid(True)

        for obst in obsts:
            obst_x, obst_y = obst._poly.exterior.xy
            ax.fill(obst_x, obst_y, color="red", alpha=0.5)

        init = spline.calc_kstate(0.0)
        goal = spline.calc_kstate(spline.length)

        init_trc = init.calc_trc_poly(tcfg)
        init_trl = init.calc_trl_poly(tcfg)
        goal_trc = goal.calc_trc_poly(tcfg)
        goal_trl = goal.calc_trl_poly(tcfg)

        init_trc_x, init_trc_y = init_trc.exterior.xy
        init_trl_x, init_trl_y = init_trl.exterior.xy
        goal_trc_x, goal_trc_y = goal_trc.exterior.xy
        goal_trl_x, goal_trl_y = goal_trl.exterior.xy

        ax.fill(init_trc_x, init_trc_y, color="magenta", alpha=0.5)
        ax.fill(init_trl_x, init_trl_y, color="cyan", alpha=0.5)
        ax.fill(goal_trc_x, goal_trc_y, color="magenta", alpha=0.5)
        ax.fill(goal_trl_x, goal_trl_y, color="cyan", alpha=0.5)

        ss = np.arange(0, spline.length, step=3.0)
        for s in ss:
            kstate = spline.calc_kstate(s)
            trc_poly = kstate.calc_trc_poly(tcfg)
            trl_poly = kstate.calc_trl_poly(tcfg)
            trc_x, trc_y = trc_poly.exterior.xy
            trl_x, trl_y = trl_poly.exterior.xy
            ax.plot(trc_x, trc_y, color="black", lw=0.4, ls="--")
            ax.plot(trl_x, trl_y, color="gray", lw=0.8, ls="--")

        xs = spline.xs
        ys = spline.ys
        ax.plot(xs, ys, color="blue" if spline.dir > 0 else "red", lw=2.0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        return buf.getvalue()
