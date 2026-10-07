#!/usr/bin/env python3
"""Drive the optimized racing line in the FULL vehicle physics.

A classical controller (pure-pursuit steering + preview braking speed
control) follows the oracle line from supra/raceline.py for one flying lap
of the real-elevation Nordschleife, with the complete 4-wheel tyre model,
active aero, hybrid ledger, crest/airborne physics — everything training
uses. This is the executability proof behind the oracle number: the oracle
assumes a perfect driver; whatever completes here is a real, driven,
footprint-checked lap in the simulator.

    python3 tools/drive_raceline_lap.py --car porsche_919evo \
        --line raceline_919_nordschleife_v1.npz --sf 0.93
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supra.config import get_car  # noqa: E402
from supra.physics import Vehicle, Controls  # noqa: E402
from supra.track import named_track  # noqa: E402


def drive_lap(car_name: str, line_path: str, sf: float = 0.93,
              dt: float = 1.0 / 120.0, verbose: bool = True) -> dict:
    trk = named_track("nordschleife")
    line = np.load(line_path)
    P = line["points"]          # (n,2) optimized line
    vref = line["v"] * float(sf)
    ds = line["ds"]
    n = len(P)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])[:n]
    total_len = float(np.sum(ds))
    tangents = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0))
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1), 1e-9)[:, None]
    kline = line["curvature"]

    spec = get_car(car_name)

    # Crest-hold cap: the oracle profile knows grade but not vertical
    # curvature, so over Flugplatz-class crests it may carry speed the road
    # cannot hold down. Cap the reference so the car keeps >=35% wheel load
    # (downforce included, in its LOW-lift state to be conservative):
    #   1 + (vcurv + q_low) v^2 / g >= 0.35  ->  v <= sqrt(0.65 g / -(vcurv+q_low))
    q_low = (0.5 * float(spec.air_density) * float(spec.downforce_ClA)
             * float(getattr(spec, "aero_lowlift_factor", 1.0)
                     if getattr(spec, "aero_active", False) else 1.0)
             / float(spec.mass)) / 9.81
    trk_pre = named_track("nordschleife")
    vcurv_line = np.array([trk_pre.frame(float(x), float(y))["vcurv"]
                           for x, y in P])
    hold = vcurv_line / 9.81 + q_low
    crest_cap = np.where(hold < -1e-9,
                         np.sqrt(0.65 / np.maximum(-hold, 1e-9)),
                         1e9)
    vref = np.minimum(vref, crest_cap)

    veh = Vehicle(spec)
    start_yaw = math.atan2(tangents[0, 1], tangents[0, 0])
    veh.reset(float(P[0, 0]), float(P[0, 1]), start_yaw)
    veh.vx = float(vref[0])
    veh.gear = min(5, len(spec.gear_ratios))

    # physics-consistent braking preview: at each speed the deceleration is
    # the LESSER of brake authority (torque) and tyre grip with downforce —
    # at low speed the tyres, not the brakes, are the limit, and planning
    # with the wrong one sends the car past the apex (measured: spin at
    # Hatzenbach with a constant-authority preview).
    brake_auth = float(spec.max_brake_torque) / float(spec.wheel_radius) / float(spec.mass)
    mu = float(spec.mu)
    q_hi = 0.5 * float(spec.air_density) * float(spec.downforce_ClA) / float(spec.mass)

    def a_brake(v_here: float) -> float:
        tyre = 0.85 * mu * (9.81 + q_hi * v_here * v_here)
        return 0.88 * min(brake_auth, tyre)

    lat_cap0 = 0.92 * mu * 9.81      # low-speed lateral budget for throttle cap

    idx = 0                     # moving nearest-point pointer
    progress = 0.0              # monotonic distance along the line
    first_fail = None
    dnf_reason = None
    t = 0.0
    off_seconds = 0.0
    foot_off_seconds = 0.0
    air_seconds = 0.0
    vmin = 1e9
    vmax = 0.0
    max_steps = int(900.0 / dt)
    horizon_m = 450.0
    hpts = int(horizon_m / max(np.mean(ds), 1e-6))

    for step in range(max_steps):
        # --- localization: advance the pointer to the nearest line point ---
        for _ in range(64):
            nxt = (idx + 1) % n
            if (np.linalg.norm(P[nxt] - [veh.x, veh.y])
                    < np.linalg.norm(P[idx] - [veh.x, veh.y])):
                d_adv = ds[idx]
                idx = nxt
                progress += d_adv
            else:
                break
        # stale pointer (car pushed wide/ahead): re-acquire in a forward
        # window only — never across the track's near-self-crossing sections
        if np.linalg.norm(P[idx] - [veh.x, veh.y]) > 12.0:
            w0 = idx
            window = [(w0 + kk) % n for kk in range(0, 400)]
            dists = np.linalg.norm(P[window] - [veh.x, veh.y], axis=1)
            best = int(np.argmin(dists))
            for kk in range(best):
                progress += ds[(w0 + kk) % n]
            idx = window[best]
        if progress >= total_len:
            break

        # --- road under the car (elevation/bank/crest exactly like training) ---
        fr = trk.frame(veh.x, veh.y)
        veh.surface_grip = 0.5 if fr["off_track"] else 1.0
        veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
        if fr["off_track"]:
            off_seconds += dt
        if step % 4 == 0:                       # 30 Hz footprint audit
            try:
                if any(trk.frame(float(x), float(y))["off_track"]
                       for x, y in veh.get_obb()):
                    foot_off_seconds += 4 * dt
                    if first_fail is None:
                        first_fail = {"t": round(t, 1),
                                      "s_km": round(progress / 1000.0, 2),
                                      "v_kmh": round(veh.speed * 3.6, 0)}
            except Exception:
                pass
        if veh.airborne:
            air_seconds += dt
        if step > 240 and veh.speed < 3.0:      # spun/stalled: honest DNF
            dnf_reason = "spun_or_stalled"
            break
        if foot_off_seconds > 6.0:              # hopeless: abort, report where
            dnf_reason = "footprint_violations"
            break

        v = max(veh.speed, 0.1)
        vmin = min(vmin, v) if step > 60 else vmin
        vmax = max(vmax, v)

        # --- steering: pure pursuit on the line ---
        L = float(np.clip(0.55 * v, 10.0, 85.0))
        j = idx
        acc = 0.0
        while acc < L:
            acc += ds[j % n]
            j += 1
        target = P[j % n]
        dx, dy = target[0] - veh.x, target[1] - veh.y
        chord = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - veh.yaw
        ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
        kappa_pp = 2.0 * math.sin(ang) / max(chord, 1e-6)
        # Stanley cross-track term: signed lateral error to the line keeps the
        # car ON the line through long high-g corners where pure pursuit
        # alone converges lazily and cuts inside.
        rel = np.array([veh.x, veh.y]) - P[idx]
        tvec = tangents[idx]
        e_lat = float(tvec[0] * rel[1] - tvec[1] * rel[0])   # + = left of line
        # Stanley gain fades with speed: tight tracking in the slow stuff,
        # stability at 300 km/h where a meter of error must not saw the wheel
        k_st = 0.9 * (22.0 / (v + 22.0))
        wheel = (math.atan(float(spec.wheelbase) * kappa_pp)
                 - math.atan(k_st * e_lat / (v + 4.0)))
        steer = float(np.clip(wheel / math.radians(spec.steer_angle_max_deg),
                              -1.0, 1.0))
        if veh.airborne:
            # in the air: hold near-straight, no inputs that spike on landing
            steer *= 0.25

        # --- speed: ANTICIPATORY braking. A reactive tracker starts braking
        # only once it is already above the falling reference — at 45 m/s^2
        # zones that is ~14 m/s hot into the apex (measured at Hatzenbach).
        # Instead, scan the preview for the maximum REQUIRED deceleration to
        # meet every upcoming reference point, and feed the brake forward as
        # the ratio of required to available deceleration.
        a_need_max = -1e9
        acc = 0.0
        for jj in range(1, hpts):
            k2 = (idx + jj) % n
            acc += float(ds[(idx + jj - 1) % n])
            dv2 = v * v - float(vref[k2]) ** 2
            if dv2 > 0.0:
                a_need_max = max(a_need_max, dv2 / (2.0 * acc))
        a_avail = a_brake(v)
        if a_need_max > 0.12 * a_avail:
            brake = float(np.clip(a_need_max / max(a_avail, 1e-6), 0.0, 1.0))
            throttle = 0.0
        else:
            err = float(vref[idx]) - v
            brake = 0.0
            throttle = float(np.clip(0.55 * err, 0.0, 1.0))
            # friction-circle cap: don't demand drive force the tyres can't
            # spare mid-corner (lateral demand from the LINE curvature here)
            lat_cap = 0.92 * mu * (9.81 + q_hi * v * v)
            lat_used = v * v * abs(float(kline[idx]))
            room = max(0.0, 1.0 - (lat_used / max(lat_cap, 1e-6)) ** 2)
            throttle = min(throttle, 0.15 + 0.85 * math.sqrt(room))

        # --- gears: race thresholds ---
        if veh.rpm > 0.965 * spec.redline_rpm and veh.gear < len(spec.gear_ratios):
            veh.gear += 1
        elif veh.rpm < 0.42 * spec.redline_rpm and veh.gear > 1:
            veh.gear -= 1

        veh.step(Controls(steer=steer, throttle=throttle, brake=brake), dt)
        t += dt

        if verbose and step % int(30.0 / dt) == 0 and step:
            print(f"  t={t:6.1f}s  s={progress/1000.0:6.2f} km  v={v*3.6:5.0f} km/h  "
                  f"off={off_seconds:.1f}s", flush=True)

    finished = progress >= total_len
    result = {
        "car": car_name,
        "line": str(line_path),
        "safety_factor": sf,
        "finished": bool(finished),
        "lap_time_s": round(t, 2) if finished else None,
        "progress_m": round(progress, 1),
        "offtrack_seconds": round(off_seconds, 2),
        "footprint_offtrack_seconds": round(foot_off_seconds, 2),
        "airborne_seconds": round(air_seconds, 2),
        "clean": bool(finished and foot_off_seconds == 0.0),
        "v_min_kmh": round(vmin * 3.6, 1),
        "v_max_kmh": round(vmax * 3.6, 1),
        "hybrid_soc_end_kj": round(getattr(veh, "hybrid_soc_kj", 0.0), 1),
        "first_footprint_violation": first_fail,
        "dnf_reason": dnf_reason,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", default="porsche_919evo")
    ap.add_argument("--line", default="raceline_919_nordschleife_v1.npz")
    ap.add_argument("--sf", type=float, default=0.93)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = drive_lap(args.car, args.line, args.sf)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
