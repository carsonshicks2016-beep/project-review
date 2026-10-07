"""Physics acceptance checks for cinematic chase fidelity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .city import UrbanGrid
from .config import get_car
from .physics import Controls, Vehicle
from .sim import HeistSim


def run_physics_validation(
    *,
    seed: int = 11,
    turn_speed: float = 24.0,
    turn_steps: int = 180,
    trace_steps: int = 360,
    long_steps: int = 1200,
    out: str | Path | None = None,
) -> dict[str, Any]:
    checks = [
        _check_deterministic_trace(seed=seed, steps=trace_steps),
        _check_handbrake_asymmetry(speed=turn_speed, steps=turn_steps),
        _check_braking_stability(),
        _check_collision_recovery(seed=seed),
        _check_long_run_stability(seed=seed, steps=long_steps),
    ]
    passed = all(check["passed"] for check in checks)
    manifest = {
        "version": 1,
        "seed": seed,
        "passed": passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks": len(checks),
        "score": sum(int(check["passed"]) for check in checks) / max(1, len(checks)),
        "car_specs": {
            name: _car_spec_summary(name)
            for name in ("evader", "pursuer")
        },
        "results": checks,
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _check_deterministic_trace(*, seed: int, steps: int) -> dict[str, Any]:
    left = HeistSim(seed=seed)
    right = HeistSim(seed=seed)
    for _ in range(steps):
        left.step()
        right.step()
    left_state = _sim_state(left)
    right_state = _sim_state(right)
    max_delta = float(np.max(np.abs(left_state - right_state)))
    passed = max_delta <= 1e-9 and left.channel.jamming_budget == right.channel.jamming_budget
    return _check(
        "deterministic_trace",
        passed,
        metrics={
            "steps": steps,
            "max_state_delta": max_delta,
            "left_jamming_budget": left.channel.jamming_budget,
            "right_jamming_budget": right.channel.jamming_budget,
        },
        thresholds={"max_state_delta": 1e-9},
    )


def _check_handbrake_asymmetry(*, speed: float, steps: int) -> dict[str, Any]:
    evader = _run_turn("evader", speed=speed, steps=steps)
    pursuer = _run_turn("pursuer", speed=speed, steps=steps)
    yaw_ratio = evader["abs_yaw_deg"] / max(1e-6, pursuer["abs_yaw_deg"])
    radius_ratio = pursuer["avg_yaw_radius"] / max(1e-6, evader["avg_yaw_radius"])
    speed_ratio = evader["final_speed"] / max(1e-6, pursuer["final_speed"])
    passed = (
        yaw_ratio >= 1.08
        and radius_ratio >= 1.35
        and speed_ratio <= 0.65
        and evader["max_slip_angle"] >= 0.45
        and evader["all_finite"]
        and pursuer["all_finite"]
    )
    return _check(
        "handbrake_asymmetry",
        passed,
        metrics={
            "entry_speed": speed,
            "steps": steps,
            "evader": evader,
            "pursuer": pursuer,
            "yaw_ratio": float(yaw_ratio),
            "radius_ratio": float(radius_ratio),
            "speed_ratio": float(speed_ratio),
        },
        thresholds={
            "min_yaw_ratio": 1.08,
            "min_radius_ratio": 1.35,
            "max_speed_ratio": 0.65,
            "min_evader_slip_angle_rad": 0.45,
        },
    )


def _check_braking_stability() -> dict[str, Any]:
    initial_speed = 36.0
    veh = Vehicle(get_car("pursuer"))
    veh.reset(speed=initial_speed)
    trace = []
    for _ in range(360):
        veh.step(Controls(steer=0.15, throttle=0.0, brake=1.0, clutch=1.0))
        trace.append([veh.x, veh.y, veh.yaw, veh.speed, float(np.max(veh.wheel_grip))])
    values = np.asarray(trace, dtype=float)
    passed = bool(
        np.isfinite(values).all()
        and veh.speed <= initial_speed * 0.62
        and float(np.max(values[:, 3])) <= initial_speed * 1.04
        and float(np.max(values[:, 4])) <= 1.05
    )
    return _check(
        "hard_braking_stability",
        passed,
        metrics={
            "initial_speed": initial_speed,
            "final_speed": float(veh.speed),
            "max_speed": float(np.max(values[:, 3])),
            "max_wheel_grip": float(np.max(values[:, 4])),
            "all_finite": bool(np.isfinite(values).all()),
        },
        thresholds={
            "max_final_speed": initial_speed * 0.62,
            "max_speed_growth": initial_speed * 1.04,
            "max_wheel_grip": 1.05,
        },
    )


def _check_collision_recovery(*, seed: int) -> dict[str, Any]:
    city = UrbanGrid(seed=seed)
    building = city.buildings[0]
    veh = Vehicle(get_car("evader"))
    veh.reset(
        x=(building.x0 + building.x1) / 2.0,
        y=(building.y0 + building.y1) / 2.0,
        speed=30.0,
    )
    initial_speed = veh.speed
    impact = city.resolve_vehicle(veh, radius=2.25)
    values = np.asarray([veh.x, veh.y, veh.yaw, veh.vx, veh.vy, veh.speed, impact], dtype=float)
    clear = city.building_at(veh.x, veh.y, pad=2.25) is None
    passed = bool(np.isfinite(values).all() and clear and impact > 0.0 and veh.speed < initial_speed * 0.25)
    return _check(
        "collision_recovery",
        passed,
        metrics={
            "impact": float(impact),
            "initial_speed": float(initial_speed),
            "final_speed": float(veh.speed),
            "clear_of_padded_building": bool(clear),
            "all_finite": bool(np.isfinite(values).all()),
            "x": float(veh.x),
            "y": float(veh.y),
        },
        thresholds={
            "min_impact": 0.0,
            "max_final_speed_ratio": 0.25,
            "requires_clear_of_padded_building": True,
        },
    )


def _check_long_run_stability(*, seed: int, steps: int) -> dict[str, Any]:
    sim = HeistSim(seed=seed, reset_on_capture=False)
    max_speed = 0.0
    max_impact = 0.0
    all_finite = True
    for _ in range(steps):
        sim.step()
        values = _sim_state(sim)
        all_finite = all_finite and bool(np.isfinite(values).all())
        max_speed = max(max_speed, max(agent.vehicle.speed for agent in sim.agents))
        max_impact = max(max_impact, sim.last_collision_impact, sim.impact_energy)
    passed = bool(all_finite and max_speed <= 75.0 and max_impact <= 90.0)
    return _check(
        "long_run_stability",
        passed,
        metrics={
            "steps": steps,
            "max_speed": float(max_speed),
            "max_impact": float(max_impact),
            "all_finite": bool(all_finite),
            "captures": int(sim.captures),
            "waypoints_hit": int(sim.waypoints_hit),
        },
        thresholds={"max_speed": 75.0, "max_impact": 90.0},
    )


def _run_turn(car: str, *, speed: float, steps: int) -> dict[str, Any]:
    veh = Vehicle(get_car(car))
    veh.reset(speed=speed)
    samples = []
    radii = []
    for _ in range(steps):
        veh.step(Controls(steer=1.0, throttle=0.35, handbrake=1.0, clutch=1.0))
        samples.append([
            veh.x,
            veh.y,
            veh.yaw,
            veh.speed,
            veh.slip_angle,
            float(np.max(veh.wheel_grip)),
            veh.r,
        ])
        if abs(veh.r) > 0.05:
            radii.append(veh.speed / abs(veh.r))
    values = np.asarray(samples, dtype=float)
    return {
        "abs_yaw_deg": float(abs(np.degrees(veh.yaw))),
        "final_speed": float(veh.speed),
        "path_displacement": float(np.hypot(veh.x, veh.y)),
        "max_slip_angle": float(np.max(np.abs(values[:, 4]))),
        "max_wheel_grip": float(np.max(values[:, 5])),
        "max_yaw_rate": float(np.max(np.abs(values[:, 6]))),
        "avg_yaw_radius": float(np.mean(radii[-60:])) if radii else 999.0,
        "all_finite": bool(np.isfinite(values).all()),
    }


def _sim_state(sim: HeistSim) -> np.ndarray:
    rows = []
    for agent in sim.agents:
        veh = agent.vehicle
        rows.append([
            veh.x,
            veh.y,
            veh.yaw,
            veh.vx,
            veh.vy,
            veh.speed,
            veh.r,
            veh.slip_angle,
        ])
    return np.asarray(rows, dtype=float)


def _car_spec_summary(name: str) -> dict[str, Any]:
    spec = get_car(name)
    return {
        "name": spec.name,
        "mass": float(spec.mass),
        "mu": float(spec.mu),
        "yaw_inertia": float(spec.yaw_inertia),
        "steer_angle_max_deg": float(spec.steer_angle_max_deg),
        "steer_rate_deg_s": float(spec.steer_rate_deg_s),
        "handbrake_torque": float(spec.handbrake_torque),
        "drive_layout": spec.drive_layout,
    }


def _check(name: str, passed: bool, *, metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "metrics": metrics,
        "thresholds": thresholds,
    }
