"""Replay-frame + Observation-sense helpers for live streams.

Frames match ``packages/shared/schemas/replay.schema.json`` items (same keys as
``RallyEnv._record_frame``). Sense debug rides on the live envelope beside the
frame — not on the frame itself — because the replay schema forbids unknown
frame properties. TRAIN G2 reads ``sense`` from the packet; WATCH needs only
``frame``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rallyai.env.rally_env import RallyEnv
from rallyai.sense import Observation, SensorSuite


def build_frame(env: RallyEnv) -> dict[str, Any]:
    """One replay-schema frame from the current env state."""
    if env.frames:
        # Prefer the frame the env just recorded — identical contract path.
        return dict(env.frames[-1])

    v = env.car.vehicle
    q = env.query
    obs = getattr(env, "_last_obs", None)
    note = ""
    if obs is not None and obs.pace_note:
        note = str(obs.pace_note.get("text", ""))
    return {
        "t": round(env.time_s, 4),
        "x": round(v.x, 4),
        "y": round(v.y, 4),
        "z": round(v.z, 4),
        "yaw": round(v.yaw, 5),
        "pitch": round(v.pitch, 5),
        "roll": round(v.roll, 5),
        "v": round(v.speed, 3),
        "vx": round(v.vx, 3),
        "vy": round(v.vy, 3),
        "vz": round(v.vz, 3),
        "yr": round(v.r, 4),
        "sa": round(v.slip_angle, 4),
        "gear": int(v.gear),
        "rpm": round(v.rpm, 1),
        "boost": round(v.boost, 3),
        "in": {
            "s": round(float(env._prev_action[0]), 3),
            "t": round(float(env._prev_action[1]), 3),
            "b": round(float(env._prev_action[2]), 3),
            "h": round(float(env._prev_action[3]), 3),
        },
        "w": [
            {
                "sa": round(float(v.wheel_slip[i]), 4),
                "sr": round(float(v.wheel_sr[i]), 4),
                "fz": round(float(v.Fz[i]), 1),
                "grip": round(float(v.wheel_grip[i]), 4),
            }
            for i in range(4)
        ],
        "air": bool(v.airborne),
        "hgt": round(max(0.0, v.z - v.road_z), 4),
        "surf": q.surface,
        "mu": round(q.mu, 4),
        "s": round(q.s, 3),
        "prog": round(q.progress, 5),
        "lat": round(q.lateral, 4),
        "note": note,
    }


def observation_to_sense(obs: Observation) -> dict[str, Any]:
    """Serialize ``Observation`` debug fields for the live ``sense`` sidecar."""
    pts = np.asarray(obs.beam_points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] == 2:
        y = np.zeros((pts.shape[0], 1), dtype=np.float64)
        pts = np.concatenate([pts[:, :1], y, pts[:, 1:2]], axis=1)
    look_pts = np.asarray(obs.lookahead_points, dtype=np.float64)
    if look_pts.size == 0:
        look_list: list[list[float]] = []
    else:
        look_list = [
            [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
            for p in look_pts
        ]
    return {
        "beam_distances": [round(float(d), 4) for d in obs.beam_distances],
        "beam_points": [
            [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
            for p in pts
        ],
        "beam_kinds": [int(k) for k in obs.beam_kinds],
        "lookahead_s": [round(float(s), 4) for s in obs.lookahead_s],
        "lookahead_points": look_list,
        "pace_note": obs.pace_note,
    }


def build_sense(env: RallyEnv) -> dict[str, Any]:
    """Observation debug from the real SensorSuite result stored on the env."""
    obs = getattr(env, "_last_obs", None)
    if obs is None:
        sensors = getattr(env, "sensors", None) or SensorSuite()
        obs = sensors.observe(env.car, env.track, env.query)
    return observation_to_sense(obs)
