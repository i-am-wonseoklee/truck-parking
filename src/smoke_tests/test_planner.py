# src/truck_parking/smoke_tests/test_planner.py
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from truck_parking.core.obstacle import Obstacle
from truck_parking.core.planner import Planner, PlannerCfg
from truck_parking.core.truck import KState, TruckCfg


def _plot_polygon(
    ax, poly, *, edge_color=None, fill_color=None, alpha=0.3, lw=1.0, ls="-"
):
    x, y = poly.exterior.xy
    if fill_color is not None:
        ax.fill(x, y, color=fill_color, alpha=alpha, linewidth=0.0)
    if edge_color is not None:
        ax.plot(x, y, color=edge_color, linewidth=lw, linestyle=ls)


def main() -> None:
    cfg = yaml.safe_load(
        Path(
            "/home/wonseok/repositories/truck-parking/config/scenario/scenario_001.yaml"
        ).read_text()
    )
    tcfg = TruckCfg().from_dict(cfg["truck_cfg"])
    pcfg = PlannerCfg().from_dict(cfg["planner_cfg"])

    init = KState.from_dict(cfg["init"])
    goal = KState.from_dict(cfg["goal"])
    obsts = [Obstacle.from_dict(obst) for obst in cfg.get("obstacles", [])]

    planner = Planner(pcfg, tcfg)

    tick = time.time()
    splines = planner.plan(init, goal, obsts)
    tock = time.time()
    elapsed = tock - tick
    print(f"Planning took {elapsed:.3f} seconds.")

    if not splines:
        print("No path found.")
        return

    n = len(splines)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 5.0 * rows))
    fig.suptitle(f"Found {n} spline(s) in {elapsed:.3f} seconds.")
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, spline in enumerate(splines):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        ax.set_aspect("equal")
        ax.set_xlim(pcfg.x_min, pcfg.x_max)
        ax.set_ylim(pcfg.y_min, pcfg.y_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Spline {i}  (len={spline.length:.2f} m)")

        # Obstacles (if any).
        for obst in obsts:
            x_obst, y_obst = obst.poly.exterior.xy
            ax.fill(x_obst, y_obst, color="gray", alpha=0.8)

        # Start and goal footprints.
        _plot_polygon(
            ax, init.calc_trc_poly(tcfg), edge_color="magenta", alpha=0.0, lw=2.0
        )
        _plot_polygon(
            ax, init.calc_trl_poly(tcfg), edge_color="green", alpha=0.0, lw=2.0
        )
        _plot_polygon(
            ax, goal.calc_trc_poly(tcfg), edge_color="magenta", alpha=0.0, lw=2.0
        )
        _plot_polygon(
            ax, goal.calc_trl_poly(tcfg), edge_color="green", alpha=0.0, lw=2.0
        )

        ss_res = int(np.round(spline.length) / 1.0) + 1
        ss = np.linspace(0.0, spline.length, ss_res)

        # Draw the track (trajectory) in red.
        xs = np.empty_like(ss)
        ys = np.empty_like(ss)
        for j, s in enumerate(ss):
            ks = spline.calc_kstate(float(s))
            xs[j] = ks.x
            ys[j] = ks.y
        ax.plot(
            xs,
            ys,
            color="red" if spline.dir > 0 else "blue",
            linewidth=2.0,
            label="trace",
        )

        # Draw footprints along the spline.
        for s in ss:
            kstate = spline.calc_kstate(float(s))
            trc = kstate.calc_trc_poly(tcfg)
            trl = kstate.calc_trl_poly(tcfg)
            _plot_polygon(ax, trc, edge_color="gray", alpha=0.0, lw=0.8, ls="--")
            _plot_polygon(ax, trl, edge_color="black", alpha=0.0, lw=0.4, ls="--")

        # Draw the last footprint more prominently for this spline.
        kstate_end = spline.calc_kstate(float(spline.length))
        _plot_polygon(ax, kstate_end.calc_trc_poly(tcfg), fill_color="blue", alpha=0.5)
        _plot_polygon(ax, kstate_end.calc_trl_poly(tcfg), fill_color="cyan", alpha=0.5)

        ax.legend(loc="upper right")

    # Hide unused subplots.
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
