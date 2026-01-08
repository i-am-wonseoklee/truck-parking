# src/truck_parking/core/planner.py
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field, fields
from typing import Dict, List, Mapping, Tuple, TypeAlias

import numpy as np
import rsplan

from truck_parking.common.geometry import wrap
from truck_parking.common.guards import (
    req_bounds_ordered,
    req_finite,
    req_positive,
    req_negative,
    req_non_negative,
    req_symmetric,
)
from truck_parking.core.obstacle import Obstacle
from truck_parking.core.spline import Spline
from truck_parking.core.truck import KState, TruckCfg


EPS: float = 1e-6

_NodeKey: TypeAlias = Tuple[int, int, int, int]
_HeuristicKey: TypeAlias = Tuple[int, int, int]
_PQItem: TypeAlias = Tuple[float, _NodeKey]


@dataclass(slots=True, frozen=True)
class PlannerCfg:
    """Configuration for the planner."""

    x_min: float = -50.0  # Min X coordinate (m).
    x_max: float = 50.0  # Max X coordinate (m).
    y_min: float = -50.0  # Min Y coordinate (m).
    y_max: float = 50.0  # Max Y coordinate (m).
    phi_min: float = -0.5 * np.pi * 0.8  # Min articulated angle (rad).
    phi_max: float = 0.5 * np.pi * 0.8  # Max articulated angle (rad).
    v_min: float = -1.0  # Max reverse speed < 0 (m/s).
    v_max: float = 1.0  # Max forward speed > 0 (m/s).
    delta_min: float = 0.9 * -0.5  # Max right steering angle < 0 (rad).
    delta_max: float = 0.9 * 0.5  # Max left steering angle > 0 (rad).

    x_res: float = 1.0  # X coordinate resolution (m).
    y_res: float = 1.0  # Y coordinate resolution (m).
    psi_res: float = 2.0 * np.pi / 18.0  # Orientation resolution (rad).
    phi_res: float = 0.8 * np.pi / 9.0  # Articulated angle resolution (rad).

    dir_penalty: float = 5.0  # Penalty for changing direction (s).

    x_tol: float = 0.1  # Arrival tolerance in X (m).
    y_tol: float = 0.1  # Arrival tolerance in Y (m).
    psi_tol: float = 0.05  # Arrival tolerance in psi (rad).
    phi_tol: float = 0.05  # Arrival tolerance in phi (rad).

    ds: float = 5.0  # Length of motion primitives (m).
    dt: float = 0.5  # Time step to simulate motion forward (s).

    # If the L2 distance to the goal is less than aexp_dist, analytical expansion is attempted.
    aexp_dist: float = 20.0  # (m).

    h_boost: float = 1.0  # Heuristic cost boost factor.

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "PlannerCfg":
        """Creates a PlannerCfg from a dictionary.

        Args:
            d (Mapping[str, float]): Configuration dictionary.

        Returns:
            PlannerCfg: Created PlannerCfg object.
        """
        obj = cls(**{f.name: d[f.name] for f in fields(cls)})
        obj._validate()
        return obj

    def to_dict(self) -> Dict[str, float]:
        """Converts the PlannerCfg to a dictionary.

        Returns:
            Dict[str, float]: Configuration dictionary.
        """
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def calc_x_key(self, x: float) -> int:
        """Returns the discretized key index for the given x coordinate.

        Args:
            x (float): X coordinate.

        Returns:
            int: Discretized key index.
        """
        return int(np.floor((x - self.x_min) / self.x_res + EPS))

    def calc_y_key(self, y: float) -> int:
        """Returns the discretized key index for the given y coordinate.

        Args:
            y (float): Y coordinate.

        Returns:
            int: Discretized key index.
        """
        return int(np.floor((y - self.y_min) / self.y_res + EPS))

    def calc_psi_key(self, psi: float) -> int:
        """Returns the discretized key index for the given psi orientation.

        Args:
            psi (float): Orientation in radians.

        Returns:
            int: Discretized key index.
        """
        return int(np.floor((psi + np.pi) / self.psi_res + EPS))

    def calc_phi_key(self, phi: float) -> int:
        """Returns the discretized key index for the given phi articulated angle.

        Args:
            phi (float): Articulated angle in radians.

        Returns:
            int: Discretized key index.
        """
        return int(np.floor((phi - self.phi_min) / self.phi_res + EPS))

    def _validate(self) -> None:
        for f in fields(self):
            req_finite(f.name, float(getattr(self, f.name)))

        req_bounds_ordered("x_min", self.x_min, "x_max", self.x_max)
        req_bounds_ordered("y_min", self.y_min, "y_max", self.y_max)
        req_negative("phi_min", self.phi_min)
        req_positive("phi_max", self.phi_max)
        req_symmetric("phi_min", self.phi_min, "phi_max", self.phi_max)
        req_negative("v_min", self.v_min)
        req_positive("v_max", self.v_max)
        req_symmetric("v_min", self.v_min, "v_max", self.v_max)
        req_negative("delta_min", self.delta_min)
        req_positive("delta_max", self.delta_max)
        req_symmetric("delta_min", self.delta_min, "delta_max", self.delta_max)
        req_bounds_ordered("phi_min", self.phi_min, "phi_max", self.phi_max)
        req_bounds_ordered("v_min", self.v_min, "v_max", self.v_max)
        req_bounds_ordered("delta_min", self.delta_min, "delta_max", self.delta_max)

        req_positive("x_res", self.x_res)
        req_positive("y_res", self.y_res)
        req_positive("psi_res", self.psi_res)
        req_positive("phi_res", self.phi_res)

        req_non_negative("dir_penalty", self.dir_penalty)

        req_positive("x_tol", self.x_tol)
        req_positive("y_tol", self.y_tol)
        req_positive("psi_tol", self.psi_tol)
        req_positive("phi_tol", self.phi_tol)

        req_positive("ds", self.ds)
        req_positive("dt", self.dt)

        req_non_negative("aexp_dist", self.aexp_dist)

        req_non_negative("h_boost", self.h_boost)


@dataclass(slots=True)
class _Node:
    """Node representation for the Hybrid A* planner."""

    kstate: KState = field(default_factory=KState)

    v: float = 0.0  # Speed to reach this node (m/s).
    delta: float = 0.0  # Steering angle to reach this node (rad).

    g: float = np.inf  # Cost from start to this node.
    h: float = np.inf  # Heuristic cost from this node to goal.
    f: float = np.inf  # Total cost (g + h).

    parent: _NodeKey | None = None  # Parent node key.


class _NodeCache:
    """Cache for storing and retrieving nodes based on discretized keys."""

    def __init__(self, cfg: PlannerCfg) -> None:
        self._cfg = cfg
        self._cache: Dict[_NodeKey, _Node] = {}

    def to_key(self, x: float, y: float, psi: float, phi: float) -> _NodeKey:
        """Creates a discretized key for the given state.

        Args:
            x (float): X coordinate (m).
            y (float): Y coordinate (m).
            psi (float): Orientation (rad).
            phi (float): Articulated angle (rad).

        Returns:
            _NodeKey: Discretized key tuple.
        """
        return (
            self._cfg.calc_x_key(x),
            self._cfg.calc_y_key(y),
            self._cfg.calc_psi_key(psi),
            self._cfg.calc_phi_key(phi),
        )

    def get(self, key: _NodeKey) -> _Node:
        """Retrieves the node for the given key, creating it if it doesn't exist.

        Args:
            key (_NodeKey): Discretized key tuple.

        Returns:
            _Node: Retrieved or newly created node.
        """
        node = self._cache.get(key)
        if node is None:
            node = _Node()
            self._cache[key] = node
        return node


class _HeurCache:
    """Cache for storing and retrieving heuristic costs based on discretized keys."""

    def __init__(self, cfg: PlannerCfg, goal: KState) -> None:
        self._cfg = cfg
        self._goal = goal
        self._cache: Dict[_HeuristicKey, float] = {}

    def to_key(self, x: float, y: float, psi: float) -> _HeuristicKey:
        """Creates a discretized key for the given state.

        Args:
            x (float): X coordinate (m).
            y (float): Y coordinate (m).
            psi (float): Orientation (rad).

        Returns:
            _HeuristicKey: Discretized key tuple.
        """
        return (
            self._cfg.calc_x_key(x),
            self._cfg.calc_y_key(y),
            self._cfg.calc_psi_key(psi),
        )

    def get(self, key: _HeuristicKey, tcfg: TruckCfg) -> float:
        """Retrieves the heuristic cost for the given key, computing it if it doesn't exist.

        Args:
            key (_HeuristicKey): Discretized key tuple.
            tcfg (TruckCfg): Truck configuration.

        Returns:
            float: Retrieved or newly computed heuristic cost.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        key_x, key_y, key_psi = key
        x = float(key_x) * self._cfg.x_res + self._cfg.x_min
        y = float(key_y) * self._cfg.y_res + self._cfg.y_min
        psi = float(key_psi) * self._cfg.psi_res - np.pi

        delta = min(abs(self._cfg.delta_min), abs(self._cfg.delta_max))
        # TODO: Test with the longer wheelbase (tractor + trailer).
        radius = tcfg.L_trc / np.tan(delta)

        v = max(self._cfg.v_max, -self._cfg.v_min)
        rs = rsplan.path(
            start_pose=(x, y, psi),
            end_pose=(self._goal.x, self._goal.y, self._goal.psi),
            turn_radius=radius,
            runway_length=0.0,
            step_size=np.hypot(self._cfg.x_res, self._cfg.y_res),
        )

        cost = 0.0
        dir_prev: int | None = None
        for segment in rs.segments:
            dir_curr = segment.direction
            if dir_prev is not None and dir_curr != dir_prev:
                cost += self._cfg.dir_penalty
            dir_prev = dir_curr
            cost += abs(segment.length) / v
        self._cache[key] = cost

        return cost


class Planner:
    """Hybrid A* planner for truck parking."""

    def __init__(self, pcfg: PlannerCfg, tcfg: TruckCfg) -> None:
        self._pcfg = pcfg
        self._tcfg = tcfg
        self._nodes: _NodeCache | None = None
        self._heurs: _HeurCache | None = None

    def plan(self, init: KState, goal: KState, obsts: List[Obstacle]) -> List[Spline]:
        """Plans a path from the initial state to the goal state avoiding obstacles.

        Args:
            init (KState): Initial state which should be collision-free.
            goal (KState): Goal state.
            obsts (List[Obstacle]): List of obstacles to avoid.

        Returns:
            List[Spline]: List of splines representing the planned path.
        """
        if self._chk_collision(init, obsts) or self._chk_collision(goal, obsts):
            return []

        self._prep_plan(goal)

        vs = [self._pcfg.v_max, self._pcfg.v_min]
        deltas = [self._pcfg.delta_min, 0.0, self._pcfg.delta_max]

        key_init = self._nodes.to_key(init.x, init.y, init.psi, init.phi)
        key_heur = self._heurs.to_key(init.x, init.y, init.psi)

        node_init = self._nodes.get(key_init)
        node_init.kstate = init
        node_init.g = 0.0
        node_init.h = self._pcfg.h_boost * self._heurs.get(key_heur, self._tcfg)
        node_init.f = node_init.g + node_init.h
        node_init.parent = None

        pq: List[_PQItem] = []
        heapq.heappush(pq, (node_init.f, key_init))
        while pq:
            f_curr, key_curr = heapq.heappop(pq)
            node_curr = self._nodes.get(key_curr)
            state_curr = node_curr.kstate

            if self._chk_arrival(state_curr, goal):
                return self._bld_splines(self._btrack(key_curr), [])

            if f_curr > node_curr.f:
                continue

            oneshot = self._try_aexp(state_curr, goal, obsts)
            if oneshot:
                return self._bld_splines(self._btrack(key_curr), oneshot)

            for v, delta in itertools.product(vs, deltas):
                pred = self._pred(state_curr, v, delta, obsts, self._pcfg.ds)
                if not pred:
                    continue

                state_next = pred[-1]

                penalty = 0.0
                if node_curr.parent is not None and node_curr.v * v < 0.0:
                    penalty = self._pcfg.dir_penalty
                g_next = node_curr.g + penalty + self._pcfg.ds / abs(v)
                key_next = self._nodes.to_key(
                    state_next.x, state_next.y, state_next.psi, state_next.phi
                )

                node_next = self._nodes.get(key_next)
                if g_next < node_next.g:
                    key_heur = self._heurs.to_key(
                        state_next.x, state_next.y, state_next.psi
                    )

                    node_next.kstate = state_next
                    node_next.v = v
                    node_next.delta = delta
                    node_next.g = g_next
                    node_next.h = self._pcfg.h_boost * self._heurs.get(
                        key_heur,
                        self._tcfg,
                    )
                    node_next.f = node_next.g + node_next.h
                    node_next.parent = key_curr

                    heapq.heappush(pq, (node_next.f, key_next))

        return []

    def _prep_plan(self, goal: KState) -> None:
        self._nodes = _NodeCache(self._pcfg)
        self._heurs = _HeurCache(self._pcfg, goal)

    def _chk_arrival(self, state: KState, goal: KState) -> bool:
        x_err = abs(state.x - goal.x)
        y_err = abs(state.y - goal.y)
        psi_err = abs(wrap(state.psi - goal.psi))
        phi_err = abs(wrap(state.phi - goal.phi))
        return (
            x_err <= self._pcfg.x_tol
            and y_err <= self._pcfg.y_tol
            and psi_err <= self._pcfg.psi_tol
            and phi_err <= self._pcfg.phi_tol
        )

    def _chk_bounds(self, state: KState) -> bool:
        return (
            self._pcfg.x_min <= state.x <= self._pcfg.x_max
            and self._pcfg.y_min <= state.y <= self._pcfg.y_max
            and self._pcfg.phi_min <= state.phi <= self._pcfg.phi_max
        )

    def _chk_collision(self, state: KState, obsts: List[Obstacle]) -> bool:
        if not obsts:
            return False
        trc_poly = state.calc_trc_poly(self._tcfg)
        trl_poly = state.calc_trl_poly(self._tcfg)
        for obst in obsts:
            if obst.chk_overlap(trc_poly) or obst.chk_overlap(trl_poly):
                return True
        return False

    def _pred(
        self,
        state: KState,
        v: float,
        delta: float,
        obsts: List[Obstacle],
        dist: float,
        *,
        skip_chks: bool = False,
    ) -> List[KState]:
        states = []
        while dist > EPS:
            dt = min(self._pcfg.dt, dist / abs(v))
            state = state.step(self._tcfg, dt, v, delta)
            if not skip_chks and (
                not self._chk_bounds(state) or self._chk_collision(state, obsts)
            ):
                return []
            states.append(state)
            dist -= abs(v) * dt
        return states

    def _try_aexp(
        self,
        state: KState,
        goal: KState,
        obsts: List[Obstacle],
    ) -> List[Tuple[int, List[KState]]]:
        dist = np.hypot(state.x - goal.x, state.y - goal.y)
        if dist > self._pcfg.aexp_dist:
            return []

        delta = min(abs(self._pcfg.delta_min), abs(self._pcfg.delta_max))
        # TODO: Test with the longer wheelbase (tractor + trailer).
        radius = self._tcfg.L_trc / np.tan(delta)
        rs = rsplan.path(
            start_pose=(state.x, state.y, state.psi),
            end_pose=(goal.x, goal.y, goal.psi),
            turn_radius=radius,
            runway_length=0.0,
            step_size=self._pcfg.ds * 0.5,  # TODO: Check meaningfulness.
        )

        dir_init = 1 if rs.segments[0].direction > 0 else -1

        segs: List[Tuple[int, List[KState]]] = [(dir_init, [state])]

        for segment in rs.segments:
            v = self._pcfg.v_max
            if segment.direction < 0:
                v = self._pcfg.v_min

            delta = 0.0
            if segment.type == "left":
                delta = self._pcfg.delta_max
            elif segment.type == "right":
                delta = self._pcfg.delta_min

            pred = self._pred(segs[-1][1][-1], v, delta, obsts, abs(segment.length))
            if not pred:
                return []

            dir = 1 if v > 0 else -1
            if dir != segs[-1][0]:
                segs.append((dir, [segs[-1][1][-1]]))
            segs[-1][1].extend(pred)

        if not self._chk_arrival(segs[-1][1][-1], goal):
            return []

        return segs

    def _btrack(self, key_curr: _NodeKey) -> List[Tuple[int, List[KState]]]:
        rkeys: List[_NodeKey] = []
        key: _NodeKey | None = key_curr
        while key is not None:
            rkeys.append(key)
            key = self._nodes.get(key).parent

        keys = list(reversed(rkeys))
        if len(keys) < 2:
            return []

        dir_init = 1 if self._nodes.get(keys[1]).v > 0 else -1
        state_init = self._nodes.get(keys[0]).kstate

        segs: List[Tuple[int, List[KState]]] = [(dir_init, [state_init])]
        for i in range(1, len(keys)):
            node = self._nodes.get(keys[i])
            pred = self._pred(
                segs[-1][1][-1],
                node.v,
                node.delta,
                [],
                dist=self._pcfg.ds,
                skip_chks=True,
            )
            dir = 1 if node.v > 0 else -1
            if dir != segs[-1][0]:
                segs.append((dir, [segs[-1][1][-1]]))
            segs[-1][1].extend(pred)

        return segs

    def _bld_splines(
        self,
        segs0: List[Tuple[int, List[KState]]],
        segs1: List[Tuple[int, List[KState]]],
    ) -> List[Spline]:
        merged_segs: List[Tuple[int, List[KState]]] = []
        if not segs0:
            merged_segs = segs1
        elif not segs1:
            merged_segs = segs0
        else:
            if segs0[-1][0] == segs1[0][0]:
                merged_segs = segs0[:-1]
                merged_segs.append((segs0[-1][0], segs0[-1][1] + segs1[0][1][1:]))
                merged_segs.extend(segs1[1:])
            else:
                merged_segs = segs0 + segs1

        return [Spline(dir, seg) for dir, seg in merged_segs]
