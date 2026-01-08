# src/truck_parking/smoke_tests/test_planner.py
import matplotlib.pyplot as plt
import numpy as np

from truck_parking.core.truck import FState, KState, TruckCfg


def main():
    tcfg = TruckCfg.from_dict(
        {
            "v_min": -5.0,
            "v_max": 5.0,
            "a_min": -2.0,
            "a_max": 2.0,
            "delta_min": -0.5,
            "delta_max": 0.5,
            "w_min": -0.3,
            "w_max": 0.3,
            "L_trc": 4.0,
            "L_trl": 12.0,
            "L_f_trc": 2.0,
            "L_f_trl": 1.0,
            "L_r_trc": 2.0,
            "L_r_trl": 1.0,
            "W_trc": 3.0,
            "W_trl": 4.0,
            "rho": -1.0,
            "phi_min": -np.pi * 0.5 * 0.8,
            "phi_max": np.pi * 0.5 * 0.8,
            "tau_v": 0.263,
            "tau_delta": 0.1,
        }
    )

    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    dt = 0.5
    steps = 20

    kstate0 = KState(x=0.0, y=0.0, psi=0.0, phi=0.0)
    for _ in range(steps):
        kstate0 = kstate0.step(tcfg, v=2.0, delta=0.2, dt=dt)

    kstate1 = KState(x=0.0, y=0.0, psi=0.0, phi=0.0)
    kstate1 = kstate1.step(tcfg, v=2.0, delta=0.2, dt=dt * steps)

    x_tractor0, y_tractor0 = kstate0.calc_trc_poly(tcfg).exterior.xy
    x_trailer0, y_trailer0 = kstate0.calc_trl_poly(tcfg).exterior.xy
    axes[0].fill(x_tractor0, y_tractor0, color="blue", alpha=0.5)
    axes[0].fill(x_trailer0, y_trailer0, color="orange", alpha=0.5)

    x_tractor1, y_tractor1 = kstate1.calc_trc_poly(tcfg).exterior.xy
    x_trailer1, y_trailer1 = kstate1.calc_trl_poly(tcfg).exterior.xy
    axes[0].fill(x_tractor1, y_tractor1, color="green", alpha=0.5)
    axes[0].fill(x_trailer1, y_trailer1, color="red", alpha=0.5)

    fstate0 = FState(
        x=0.0,
        y=0.0,
        psi=2.0,
        phi=0.1,
        v=0.0,
        delta=0.0,
        a=0.0,
        w=0.0,
    )
    for _ in range(steps):
        fstate0 = fstate0.step(tcfg, v=2.0, delta=0.2, dt=dt)
    fstate1 = FState(
        x=0.0,
        y=0.0,
        psi=2.0,
        phi=0.1,
        v=0.0,
        delta=0.0,
        a=0.0,
        w=0.0,
    )
    fstate1 = fstate1.step(tcfg, v=2.0, delta=0.2, dt=dt * steps)
    x_tractor0, y_tractor0 = fstate0.kstate.calc_trc_poly(tcfg).exterior.xy
    x_trailer0, y_trailer0 = fstate0.kstate.calc_trl_poly(tcfg).exterior.xy
    axes[1].fill(x_tractor0, y_tractor0, color="blue", alpha=0.5)
    axes[1].fill(x_trailer0, y_trailer0, color="orange", alpha=0.5)
    x_tractor1, y_tractor1 = fstate1.kstate.calc_trc_poly(tcfg).exterior.xy
    x_trailer1, y_trailer1 = fstate1.kstate.calc_trl_poly(tcfg).exterior.xy
    axes[1].fill(x_tractor1, y_tractor1, color="green", alpha=0.5)
    axes[1].fill(x_trailer1, y_trailer1, color="red", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    main()
