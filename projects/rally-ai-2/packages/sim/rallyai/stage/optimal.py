"""Theoretical minimum stage time from a forward-backward speed profile.

Not a full lap-time optimiser. A lower bound on a real drive:

1. Curvature-limited speed at each centerline sample, ``v_max = sqrt(µ · g · r)``,
   adjusted for camber.
2. Forward pass limited by available longitudinal acceleration (torque curve +
   remaining friction-circle grip).
3. Backward pass limited by braking (hardware and remaining grip).
4. Integrate ``dt = ds / v`` from start to finish.

Pure function of the stage dict (and the car). **Do not** write the result into
the stage document's ``generator`` block — that block is provenance only
(``additionalProperties: false``). Compute once at ``RallyEnv`` construction and
hold it on the env.

Calibration (24 seeds/tier, finished reference-pilot runs): mean pilot/theo
ratios ≈ 1.18–1.23 across tiers; none below 1.0. See README Measured.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rallyai.physics.cars import REFERENCE_SURFACE_MU, evo_rally, get
from rallyai.physics.vendor import CarSpec
from rallyai.track import Track

G = 9.81
# Standing-start floor. Zero would make ``dt = ds / v`` blow up; a crawl speed
# is the honest lower bound for a car that has to leave the line.
V_FLOOR = 2.0


def _resolve_car(car: CarSpec | str | None) -> CarSpec:
    if car is None:
        return evo_rally()
    if isinstance(car, str):
        return get(car)
    return car


def _peak_wheel_power(car: CarSpec) -> float:
    """Peak engine power (W) after drivetrain losses."""
    best = 0.0
    for rpm, tq in car.torque_curve:
        best = max(best, float(tq) * float(rpm) * 2.0 * math.pi / 60.0)
    return best * float(car.drivetrain_efficiency)


def _geared_accel(speed: float, car: CarSpec, ratios: np.ndarray,
                  curve_rpm: np.ndarray, curve_tq: np.ndarray) -> float:
    """Best available longitudinal acceleration from any forward gear."""
    wheel_r = float(car.wheel_radius)
    wheel_w = max(float(speed), 0.1) / wheel_r
    best_force = 0.0
    cutoff = float(car.cutoff_rpm)
    for ratio in ratios:
        rpm = wheel_w * float(ratio) * 60.0 / (2.0 * math.pi)
        if rpm < car.idle_rpm * 0.75 or rpm >= cutoff:
            continue
        tq = float(np.interp(rpm, curve_rpm, curve_tq))
        force = tq * float(ratio) * float(car.drivetrain_efficiency) / wheel_r
        best_force = max(best_force, force)
    return best_force / float(car.mass)


def theoretical_minimum_profile(
    stage: dict[str, Any],
    car: CarSpec | str | None = None,
) -> dict[str, Any]:
    """Forward-backward speed envelope over the stage centerline.

    Returns ``time_s``, the speed profile ``v`` (m/s on the dense Track grid),
    ``v_max_corner`` (curvature caps before the accel/brake passes), and a few
    diagnostic scalars. Open stages: one forward pass, one backward pass — no
    wrap-around sweeps.
    """
    car = _resolve_car(car)
    track = Track(stage)

    n = len(track.s)
    if n < 2:
        raise ValueError(f"stage {track.id}: need at least 2 centerline samples")

    # Segment lengths. The last sample has no forward segment; give it a
    # sentinel so the arrays stay length-n (mirrors Supra's convention).
    ds = np.empty(n, dtype=np.float64)
    ds[:-1] = np.diff(track.s)
    ds[-1] = ds[-2] if n > 1 else track.ds

    k = np.abs(track.curvature)
    k = np.maximum(k, 1e-9)
    # Kill single-sample curvature spikes from resampling — same 3-tap soften
    # Supra uses. Without it a one-sample kink forces an unrealistically deep
    # braking scoop that the rest of the profile cannot justify.
    k = (np.roll(k, -1) + 2.0 * k + np.roll(k, 1)) / 4.0
    k[0], k[-1] = abs(track.curvature[0]) + 1e-9, abs(track.curvature[-1]) + 1e-9

    grade = np.asarray(track.grade, dtype=np.float64)
    # Peak tyre µ at each sample: car quoted against dry gravel, stage µ is
    # absolute. Matches ``surface_grip`` in physics/cars.py.
    mu = float(car.mu) * (np.asarray(track.mu, dtype=np.float64) / REFERENCE_SURFACE_MU)
    # Positive camber banks into a left-hander; positive curvature is a left
    # turn. Their product is the bank that assists the turn.
    phi = np.asarray(track.camber, dtype=np.float64) * np.sign(track.curvature)

    m = float(car.mass)
    rho = float(getattr(car, "air_density", 1.225))
    q = 0.5 * rho * float(getattr(car, "downforce_ClA", 0.0)) / m
    cd = 0.5 * rho * float(car.drag_area) / m
    crr = float(car.rolling_resistance) * G
    p_kg = _peak_wheel_power(car) / m
    ratios = np.asarray(car.gear_ratios, dtype=np.float64) * float(car.final_drive)
    wheel_r = float(car.wheel_radius)
    curve_rpm = np.asarray([p[0] for p in car.torque_curve], dtype=np.float64)
    curve_tq = np.asarray([p[1] for p in car.torque_curve], dtype=np.float64)
    brake_auth = float(car.max_brake_torque) / wheel_r / m

    # Gear-limited terminal speed (redline in top gear). Rally cars are
    # gear-limited, not drag-limited — see the evo_rally final-drive note.
    top_ratio = float(ratios[-1])
    v_gear = (float(car.redline_rpm) * 2.0 * math.pi / 60.0) * wheel_r / top_ratio
    # Drag-limited ceiling as a soft cap on top of that.
    lo, hi = 10.0, max(40.0, v_gear + 10.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if (cd * mid * mid + crr) * mid < p_kg:
            lo = mid
        else:
            hi = mid
    v_terminal = min(v_gear, 0.5 * (lo + hi))

    # 1) Cornering caps. Camber into the turn raises the lateral limit:
    #    a_lat = g · (µ cos φ + sin φ). Downforce grows the normal load.
    v = np.full(n, v_terminal, dtype=np.float64)
    for _ in range(20):
        a_lat = G * (mu * np.cos(phi) + np.sin(phi)) + mu * q * v * v
        a_lat = np.maximum(a_lat, 1e-6)
        vc = np.sqrt(a_lat / k)
        v = np.minimum(vc, v_terminal)

    v_corner = v.copy()

    # 2) Forward pass — accelerate under engine + remaining grip.
    v[0] = min(v[0], V_FLOOR)
    for i in range(n - 1):
        vi = float(v[i])
        a_cap = float(mu[i]) * (G + q * vi * vi)
        lat_used = (vi * vi * float(k[i])) / max(a_cap, 1e-9)
        room = max(0.0, 1.0 - lat_used * lat_used)
        a_grip = a_cap * math.sqrt(room)
        a_pw = min(p_kg / max(vi, 4.0),
                   _geared_accel(vi, car, ratios, curve_rpm, curve_tq))
        a = min(a_grip, a_pw) - cd * vi * vi - crr - G * float(grade[i])
        vj2 = max(V_FLOOR * V_FLOOR, vi * vi + 2.0 * a * float(ds[i]))
        v[i + 1] = min(float(v[i + 1]), math.sqrt(vj2))

    # 3) Backward pass — brake under hardware + remaining grip.
    for i in range(n - 2, -1, -1):
        vj = float(v[i + 1])
        a_cap = float(mu[i + 1]) * (G + q * vj * vj)
        lat_used = (vj * vj * float(k[i + 1])) / max(a_cap, 1e-9)
        room = max(0.0, 1.0 - lat_used * lat_used)
        a_wheel = min(a_cap * math.sqrt(room), brake_auth)
        a_br = a_wheel + cd * vj * vj + crr + G * float(grade[i + 1])
        a_br = max(a_br, 1.0)
        v[i] = min(float(v[i]), math.sqrt(vj * vj + 2.0 * a_br * float(ds[i])))

    v = np.maximum(v, V_FLOOR)

    # 4) Integrate start → finish only. Road past the lines is scenery.
    s0 = float(track.start_s)
    s1 = float(track.finish_s)
    # Per-sample contribution: ds[i] is the step from i → i+1, charged to
    # sample i's speed. Sum over segments that overlap [start, finish].
    time_s = 0.0
    for i in range(n - 1):
        a, b = float(track.s[i]), float(track.s[i + 1])
        if b <= s0 or a >= s1:
            continue
        lo_s = max(a, s0)
        hi_s = min(b, s1)
        time_s += (hi_s - lo_s) / float(v[i])

    return {
        "time_s": float(time_s),
        "v": v,
        "v_corner": v_corner,
        "v_terminal": float(v_terminal),
        "length_m": float(s1 - s0),
        "pace_ref_mps": float((s1 - s0) / max(time_s, 1e-9)),
    }


def theoretical_minimum_time(
    stage: dict[str, Any],
    car: CarSpec | str | None = None,
) -> float:
    """Lower-bound stage time in seconds. Pure function of the stage dict."""
    return float(theoretical_minimum_profile(stage, car)["time_s"])


def optimal_sector_times_s(
    stage: dict[str, Any],
    sectors: list[dict[str, Any]],
    car: CarSpec | str | None = None,
) -> list[float]:
    """Integrate the speed envelope between archetype-boundary sector ends.

    ``sectors`` entries need ``s_start`` / ``s_end`` (metres along the
    centerline). Used by the eval harness (D4).
    """
    profile = theoretical_minimum_profile(stage, car)
    track = Track(stage)
    s = np.asarray(track.s, dtype=np.float64)
    v = np.asarray(profile["v"], dtype=np.float64)
    if s.size < 2 or v.size != s.size:
        raise ValueError("optimal profile length does not match track samples")
    ds = np.diff(s)
    dt = ds / np.maximum(v[:-1], 1e-6)
    cum = np.concatenate([[0.0], np.cumsum(dt)])

    def _t_at(s_query: float) -> float:
        return float(np.interp(s_query, s, cum, left=0.0, right=float(cum[-1])))

    return [_t_at(float(sec["s_end"])) - _t_at(float(sec["s_start"])) for sec in sectors]
