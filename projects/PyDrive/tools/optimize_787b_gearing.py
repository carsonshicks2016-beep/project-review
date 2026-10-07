#!/usr/bin/env python3
"""Deterministically select a five-speed 787B ratio set for the Nordschleife.

The search uses the simulator's torque, tyre, drag and limiter constants.  It
minimises full-throttle acceleration time across the Ring speed range plus the
measured/default shift interruption, while enforcing safe post-shift RPM and
top-speed headroom.  JSON output is suitable for provenance and regression.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supra.config import mazda787b  # noqa: E402


SEED = 787055
SHIFT_SECONDS = 0.24
TARGET_SPEED = 92.0


def _torque(spec, rpm):
    return float(np.interp(rpm, [p[0] for p in spec.torque_curve],
                           [p[1] for p in spec.torque_curve]))


def score(spec, ratios, final_drive):
    ratios = np.asarray(ratios, dtype=float)
    speeds = np.linspace(12.0, TARGET_SPEED, 321)
    accel = []
    chosen = []
    rho = float(spec.air_density)
    for speed in speeds:
        best = (0.0, 1)
        for i, gear in enumerate(ratios, 1):
            total = gear * final_drive
            rpm = speed / spec.wheel_radius * total * 60.0 / (2.0 * math.pi)
            if rpm < spec.idle_rpm * 0.75 or rpm >= spec.cutoff_rpm:
                continue
            force = (_torque(spec, rpm) * total * spec.drivetrain_efficiency
                     / spec.wheel_radius)
            if force > best[0]:
                best = (force, i)
        drag = 0.5 * rho * spec.drag_area * speed * speed
        rolling = spec.rolling_resistance * spec.mass * 9.81
        accel.append((best[0] - drag - rolling) / spec.mass)
        chosen.append(best[1])
    accel = np.asarray(accel)
    if np.any(accel <= 0.05):
        return None
    dv = float(speeds[1] - speeds[0])
    accel_time = float(np.sum(dv / accel))
    shifts = sum(a != b for a, b in zip(chosen, chosen[1:]))
    retention = ratios[1:] / ratios[:-1]
    if np.any(retention < 0.63) or np.any(retention > 0.84):
        return None
    top_rpm = (TARGET_SPEED / spec.wheel_radius * ratios[-1] * final_drive
               * 60.0 / (2.0 * math.pi))
    if not 0.90 * spec.redline_rpm <= top_rpm <= 0.985 * spec.redline_rpm:
        return None
    return {
        "score_seconds": accel_time + shifts * SHIFT_SECONDS,
        "acceleration_seconds": accel_time,
        "shifts": shifts,
        "top_rpm_at_target": top_rpm,
        "retention": retention.tolist(),
    }


def optimise(samples=30000):
    spec = mazda787b()
    rng = np.random.default_rng(SEED)
    candidates = []
    baselines = [[2.85, 2.00, 1.55, 1.25, 0.85],
                 [2.85, 2.00, 1.55, 1.20, 0.85]]
    pool = [(b, 3.50, "baseline") for b in baselines]
    for _ in range(int(samples)):
        fd = float(rng.uniform(3.15, 3.85))
        top = (spec.redline_rpm * 2.0 * math.pi * spec.wheel_radius
               / (60.0 * TARGET_SPEED * fd) * rng.uniform(0.90, 0.985))
        retention = rng.uniform(0.64, 0.83, 4)
        ratios = [top]
        for keep in retention[::-1]:
            ratios.append(ratios[-1] / keep)
        ratios = list(reversed(ratios))
        if not 2.5 <= ratios[0] <= 3.4:
            continue
        pool.append((ratios, fd, "search"))
    for ratios, fd, source in pool:
        result = score(spec, ratios, fd)
        if result:
            candidates.append({"ratios": [round(float(x), 5) for x in ratios],
                               "final_drive": round(float(fd), 5),
                               "source": source, **result})
    candidates.sort(key=lambda x: (x["score_seconds"], x["shifts"]))
    if not candidates:
        raise RuntimeError("no feasible five-speed candidates")
    return {
        "schema": "mazda787b-5spd-ring-optimizer-v1",
        "seed": SEED,
        "samples": int(samples),
        "target_speed_mps": TARGET_SPEED,
        "shift_seconds": SHIFT_SECONDS,
        "selected": candidates[0],
        "top_candidates": candidates[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = optimise(args.samples)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
