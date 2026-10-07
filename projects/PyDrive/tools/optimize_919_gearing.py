#!/usr/bin/env python3
"""Deterministically select a seven-speed Porsche 919 Evo ratio set for the
Nordschleife.

Mirrors tools/optimize_787b_gearing.py's methodology (minimise full-throttle
acceleration time across the Ring speed range plus shift interruption, subject
to safe post-shift RPM, adjacent-gear retention, and top-speed headroom) but
for the 919's seven-speed sequential box, 8500 rpm limiter, and higher terminal
speed. JSON output is suitable for provenance and regression.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supra.config import porsche_919evo  # noqa: E402


SEED = 919055
SHIFT_SECONDS = 0.18            # paddle-shift LMP1 box: near-seamless
TARGET_SPEED = 102.0           # 7th sized here (~367 km/h): just under the
                               # 369.4 km/h record-run top speed in low-drag mode
N_GEARS = 7
RETENTION_LO = 0.74            # 7 gears sit closer than the 787B's 5
RETENTION_HI = 0.90


def _torque(spec, rpm):
    return float(np.interp(rpm, [p[0] for p in spec.torque_curve],
                           [p[1] for p in spec.torque_curve]))


def score(spec, ratios, final_drive):
    ratios = np.asarray(ratios, dtype=float)
    speeds = np.linspace(12.0, TARGET_SPEED, 431)
    accel = []
    chosen = []
    rho = float(spec.air_density)
    # top-gear sizing happens at V-max, which the Evo only reaches in its
    # DRS-style low-drag state — use that CdA when the car has active aero
    drag_area = float(spec.drag_area) * (float(getattr(spec, "aero_lowdrag_factor", 1.0))
                                         if getattr(spec, "aero_active", False) else 1.0)
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
        
        if getattr(spec, "is_ttr_hybrid", False):
            mgu_ratio = getattr(spec, "mgu_gear_ratio", 5.5)
            mgu_power = float(getattr(spec, "hybrid_mgu_power_w", 0.0))
            if mgu_power > 0.0:
                mgu_omega = max(speed / max(spec.wheel_radius, 0.1), 1.0) * mgu_ratio
                deploy_factor = 1.0
                mgu_tq = (mgu_power / mgu_omega) * deploy_factor
                mgu_force = mgu_tq * mgu_ratio * spec.drivetrain_efficiency / spec.wheel_radius
                best = (best[0] + mgu_force, best[1])

        drag = 0.5 * rho * drag_area * speed * speed
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
    if np.any(retention < RETENTION_LO) or np.any(retention > RETENTION_HI):
        return None
    top_rpm = (TARGET_SPEED / spec.wheel_radius * ratios[-1] * final_drive
               * 60.0 / (2.0 * math.pi))
    if not 0.90 * spec.redline_rpm <= top_rpm <= 0.985 * spec.redline_rpm:
        return None
    # every ratio must be reachable somewhere as the best gear (no dead ratio)
    if len(set(chosen)) < N_GEARS:
        return None
    return {
        "score_seconds": accel_time + shifts * SHIFT_SECONDS,
        "acceleration_seconds": accel_time,
        "shifts": shifts,
        "top_rpm_at_target": top_rpm,
        "retention": retention.tolist(),
    }


def optimise(samples=60000):
    spec = porsche_919evo()
    rng = np.random.default_rng(SEED)
    candidates = []
    for _ in range(int(samples)):
        fd = float(rng.uniform(3.10, 4.10))
        top = (spec.redline_rpm * 2.0 * math.pi * spec.wheel_radius
               / (60.0 * TARGET_SPEED * fd) * rng.uniform(0.90, 0.985))
        retention = rng.uniform(RETENTION_LO + 0.01, RETENTION_HI - 0.01, N_GEARS - 1)
        ratios = [top]
        for keep in retention[::-1]:
            ratios.append(ratios[-1] / keep)
        ratios = list(reversed(ratios))
        if not 2.2 <= ratios[0] <= 3.2:
            continue
        result = score(spec, ratios, fd)
        if result:
            candidates.append({"ratios": [round(float(x), 5) for x in ratios],
                               "final_drive": round(float(fd), 5),
                               "source": "search", **result})
    candidates.sort(key=lambda x: (x["score_seconds"], x["shifts"]))
    if not candidates:
        raise RuntimeError("no feasible seven-speed candidates")
    return {
        "schema": "porsche919evo-7spd-ring-optimizer-v2",
        "seed": SEED,
        "samples": int(samples),
        "target_speed_mps": TARGET_SPEED,
        "shift_seconds": SHIFT_SECONDS,
        "n_gears": N_GEARS,
        "retention_band": [RETENTION_LO, RETENTION_HI],
        "selected": candidates[0],
        "top_candidates": candidates[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = optimise(args.samples)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps({"selected": report["selected"]}, indent=2))


if __name__ == "__main__":
    main()
