"""Relative orbital motion in the target's LVLH (Hill) frame.

Axes, with the target on a circular orbit of mean motion n:
    x -> radial      (R-bar, +x points away from Earth)
    y -> along-track (V-bar, +y points along the target's velocity)
    z -> cross-track (H-bar, orbit normal)

Clohessy-Wiltshire linearised equations of relative motion:
    x_ddot =  3 n^2 x + 2 n y_dot + ax
    y_ddot =           -2 n x_dot + ay
    z_ddot =   -n^2 z             + az

Valid while the separation is small compared with the orbit radius, which
holds comfortably for the sub-kilometre proximity operations modelled here.
"""

from __future__ import annotations

import numpy as np

MU_EARTH = 3.986004418e14  # m^3 / s^2
R_EARTH = 6.378137e6  # m


def mean_motion(altitude_m: float) -> float:
    """Mean motion of a circular orbit at `altitude_m` above Earth's surface."""
    a = R_EARTH + altitude_m
    return float(np.sqrt(MU_EARTH / a**3))


def cw_derivative(state: np.ndarray, accel: np.ndarray, n: float) -> np.ndarray:
    """Time derivative of [x, y, z, xd, yd, zd] under CW dynamics."""
    x, _, z, xd, yd, zd = state
    return np.array(
        [
            xd,
            yd,
            zd,
            3.0 * n * n * x + 2.0 * n * yd + accel[0],
            -2.0 * n * xd + accel[1],
            -n * n * z + accel[2],
        ]
    )


def rk4_step(state: np.ndarray, accel: np.ndarray, n: float, dt: float) -> np.ndarray:
    """One RK4 step with the control acceleration held constant over `dt`."""
    k1 = cw_derivative(state, accel, n)
    k2 = cw_derivative(state + 0.5 * dt * k1, accel, n)
    k3 = cw_derivative(state + 0.5 * dt * k2, accel, n)
    k4 = cw_derivative(state + dt * k3, accel, n)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def cw_state_transition(n: float, t: float) -> np.ndarray:
    """Closed-form CW state transition matrix, used for the analytic baseline."""
    s, c = np.sin(n * t), np.cos(n * t)
    return np.array(
        [
            [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
            [6 * (s - n * t), 1, 0, -2 * (1 - c) / n, (4 * s - 3 * n * t) / n, 0],
            [0, 0, c, 0, 0, s / n],
            [3 * n * s, 0, 0, c, 2 * s, 0],
            [-6 * n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
            [0, 0, -n * s, 0, 0, c],
        ]
    )
