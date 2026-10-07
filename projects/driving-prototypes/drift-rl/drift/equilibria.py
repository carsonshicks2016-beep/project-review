"""Solve for steady-state drift equilibria of the vehicle model.

A 'drift equilibrium' is a cornering state where the rear axle is well past its
friction peak yet every derivative is zero: the car holds a constant radius,
constant speed and constant (large) slip angle. These points are open-loop
unstable, which is why drifting is a skill, but their existence is what makes
the task learnable at all. The solutions here are also used as feed-forward
targets by the scripted pilot in ``drift.pilot``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import fsolve

from drift.config import CarConfig
from drift.physics import G, tire_forces

_MAXS = 1e3


@dataclass(frozen=True)
class Equilibrium:
    speed: float
    slip_rear: float      # rad, signed
    beta: float           # chassis sideslip, rad
    yaw_rate: float       # rad/s
    steer: float          # rad, signed (negative of the turn = opposite lock)
    throttle: float       # 0..1
    kappa: float          # rear slip ratio
    radius: float         # m, cornering radius
    residual: float
    drivable: bool = False   # solved *and* inside the car's control authority

    @property
    def ok(self) -> bool:
        return self.residual < 1e-6 and self.drivable


def _residuals(z, cfg: CarConfig, speed: float, ar: float):
    beta, omega, steer, throttle = z
    L = cfg.a + cfg.b
    u, v = speed * math.cos(beta), speed * math.sin(beta)
    # Rear slip angle constraint pins the yaw rate.
    r = (v - u * math.tan(ar)) / cfg.b

    ux = max(abs(u), 2.0)
    af = math.atan2(v + cfg.a * r, ux) - steer
    kappa = (omega * cfg.wheel_radius - u) / ux

    # Load transfer depends on the specific force Fx/m, which depends on the
    # loads: iterate to a fixed point. Must match drift.physics exactly, so it
    # uses Fx/m and not u_dot (which carries the v*r frame term).
    spec = 0.0
    for _ in range(6):
        transfer = cfg.mass * spec * cfg.cg_height / L
        lim = 0.75 * cfg.mass * G
        transfer = max(-lim, min(lim, transfer))
        fz_f = max(0.0, cfg.mass * G * cfg.b / L - transfer)
        fz_r = max(0.0, cfg.mass * G * cfg.a / L + transfer)
        fxf, fyf = tire_forces(af, 0.0, fz_f, cfg.front_tire)
        fxr, fyr = tire_forces(ar, kappa, fz_r, cfg.rear_tire)
        drag = cfg.drag_coef * u * abs(u) + cfg.roll_resist * cfg.mass * G * math.copysign(1.0, u)
        sd, cd = math.sin(steer), math.cos(steer)
        fx = fxf * cd - fyf * sd + fxr - drag
        spec = fx / cfg.mass

    fy = fxf * sd + fyf * cd + fyr
    mz = cfg.a * (fxf * sd + fyf * cd) - cfg.b * fyr

    wheel_w = max(abs(omega), 1.0)
    t_cap = min(cfg.drive_torque, cfg.drive_power / wheel_w)
    t_drive = throttle * t_cap if throttle >= 0 else throttle * cfg.brake_torque

    return [
        spec + v * r,                                  # u_dot
        fy / cfg.mass - u * r,                         # v_dot
        mz / cfg.inertia,                              # r_dot
        (t_drive - fxr * cfg.wheel_radius) / cfg.wheel_inertia,  # omega_dot
    ]


def _plausible(eq: Equilibrium, cfg: CarConfig) -> bool:
    """Reject numerical solutions the car could not actually be driven to."""
    return (eq.residual < 1e-6
            and abs(eq.steer) <= cfg.max_steer
            and -0.2 <= eq.throttle <= 1.0
            and 0.0 <= eq.kappa <= 3.0
            and math.copysign(1.0, eq.yaw_rate) == -math.copysign(1.0, eq.slip_rear)
            and math.copysign(1.0, eq.steer) == math.copysign(1.0, eq.slip_rear))


def _attempt(guess, cfg, speed, slip_rear) -> Equilibrium:
    z, info, _flag, _msg = fsolve(_residuals, np.array(guess), args=(cfg, speed, slip_rear),
                                  full_output=True, xtol=1e-12, maxfev=4000)
    beta, omega, steer, throttle = z
    res = float(np.abs(info["fvec"]).max())
    u, v = speed * math.cos(beta), speed * math.sin(beta)
    r = (v - u * math.tan(slip_rear)) / cfg.b
    kappa = (omega * cfg.wheel_radius - u) / max(abs(u), 2.0)
    radius = abs(speed / r) if abs(r) > 1e-6 else float("inf")
    eq = Equilibrium(speed, slip_rear, beta, r, steer, throttle, kappa, radius, res)
    return replace(eq, drivable=_plausible(eq, cfg))


def solve(speed: float, slip_rear: float, cfg: CarConfig | None = None,
          guess: tuple[float, float, float, float] | None = None) -> Equilibrium:
    """Find a drivable equilibrium at a given speed and (signed) rear slip angle.

    fsolve only converges from inside a fairly narrow basin, so try the warm
    start first and then fan out over a small grid of physically sensible ones.
    """
    cfg = cfg or CarConfig()
    d = -math.copysign(1.0, slip_rear)          # +1 for a left-hand drift
    guesses = [guess] if guess is not None else []
    for bf in (0.75, 0.85, 0.95):
        for kap in (0.25, 0.5, 0.9):
            for st in (0.25, 0.45, 0.65):
                guesses.append((bf * slip_rear,
                                (1.0 + kap) * speed / cfg.wheel_radius,
                                -st * d * cfg.max_steer / 0.66,
                                0.3 + 0.25 * kap))
    best = None
    for g in guesses:
        try:
            eq = _attempt(g, cfg, speed, slip_rear)
        except Exception:
            continue
        if eq.drivable:
            return eq
        if best is None or eq.residual < best.residual:
            best = eq
    return best


def sweep(speed: float, angles_deg=(15, 20, 25, 30, 35, 40, 45),
          cfg: CarConfig | None = None, left: bool = True) -> list[Equilibrium]:
    """Walk out along the slip-angle axis at one speed, warm-starting each solve."""
    cfg = cfg or CarConfig()
    sign = -1.0 if left else 1.0
    out, guess = [], None
    for a in angles_deg:
        eq = solve(speed, sign * math.radians(a), cfg, guess)
        out.append(eq)
        if eq.ok:
            guess = _as_guess(eq, cfg)
    return out


def _as_guess(eq: Equilibrium, cfg: CarConfig):
    omega = (1.0 + eq.kappa) * eq.speed / cfg.wheel_radius
    return (eq.beta, omega, eq.steer, eq.throttle)


def continuation(slip_rear: float, speeds, cfg: CarConfig | None = None) -> list[Equilibrium]:
    """Track one slip angle up through speed, using each solution to seed the next.

    Equilibria at speed are hard to find cold - the basin is narrow - but easy to
    walk to from a slower one, so the library is built by continuation.
    """
    cfg = cfg or CarConfig()
    out, guess = [], None
    for v in speeds:
        eq = solve(v, slip_rear, cfg, guess)
        out.append(eq)
        if eq.ok:
            guess = _as_guess(eq, cfg)
        else:
            break
    return out


def library(cfg: CarConfig | None = None, angles_deg=(20, 25, 30, 35, 40, 45),
            speeds=tuple(float(v) for v in range(10, 27))) -> dict:
    """Build {slip_deg: [equilibria by speed]} - the pilot's feed-forward table."""
    cfg = cfg or CarConfig()
    table = {}
    for a in angles_deg:
        eqs = [e for e in continuation(-math.radians(a), speeds, cfg) if e.ok]
        if eqs:
            table[a] = eqs
    return table


if __name__ == "__main__":
    print(f"{'slip':>6} {'beta':>7} {'yaw':>7} {'steer':>7} {'thr':>6} {'kappa':>6} {'R(m)':>7}  ok")
    for v in (12.0, 16.0, 20.0):
        print(f"-- {v:.0f} m/s " + "-" * 46)
        for eq in sweep(v):
            print(f"{math.degrees(eq.slip_rear):6.1f} {math.degrees(eq.beta):7.1f} "
                  f"{eq.yaw_rate:7.2f} {math.degrees(eq.steer):7.1f} {eq.throttle:6.2f} "
                  f"{eq.kappa:6.2f} {eq.radius:7.1f}  {'y' if eq.ok else 'n'}")
