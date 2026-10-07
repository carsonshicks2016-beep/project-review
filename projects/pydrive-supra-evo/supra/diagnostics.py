"""Deterministic checkpoint telemetry export for agent diagnosis.

This module is analysis-only: it replays saved policies and writes raw traces
without weakening the normal watch/train checkpoint guards. Old checkpoints are
adapted here by slicing/padding observations, never by rewriting the checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import PPOSpec, SensorSpec, get_car
from .physics import Vehicle
from .sensors import SensorSuite
from .track import named_list, named_track, oval, random_circuit, touge


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"{type(v).__name__} is not JSON serializable")


def _mode_vec(mode: str) -> np.ndarray:
    if mode == "hybrid":
        return np.array([1.0, 1.0], dtype=np.float32)
    if mode == "drift":
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)


def resolve_track(name: str, seed: int = 7):
    name = (name or "club").strip()
    if name in named_list():
        return named_track(name)
    if name == "oval":
        return oval()
    if name == "touge":
        return touge(seed=seed)
    if name == "random":
        return random_circuit(seed=seed)
    return named_track("club")


class DiagnosticPolicy:
    """Direct checkpoint loader for replay diagnostics.

    PPO policies return env-native actions. GA policies return their native
    steer/throttle/brake and expose a 2D longitudinal action for SupraEnv.
    """

    def __init__(self, checkpoint: str):
        self.path = os.path.abspath(checkpoint)
        self.name = os.path.basename(checkpoint)
        self.kind = "ppo" if checkpoint.endswith(".pt") else "ga"
        self.meta: dict[str, Any] = {"checkpoint": self.name}
        self.net = None
        self.norm = None
        self.brain = None
        self.mode = "race"
        self.car = "supra"
        self.obs_dim = 0
        self.sdim = 0
        self.act_dim = 0
        if self.kind == "ppo":
            self._load_ppo()
        else:
            self._load_ga()

    @property
    def legacy(self) -> bool:
        return self.kind == "ppo" and int(self.obs_dim) == 42 or self.kind == "ga" and int(self.sdim) == 40

    def _load_ppo(self):
        import torch
        from .ppo import ActorCritic, RunningNorm

        d = torch.load(self.path, map_location="cpu", weights_only=False)
        self.meta.update({k: v for k, v in d.items()
                          if k not in ("state_dict", "opt", "norm_mean", "norm_var")})
        self.mode = str(d.get("mode", "race"))
        self.car = str(d.get("car", "supra"))
        self.obs_dim = int(d["obs_dim"])
        self.sdim = int(d.get("sdim", self.obs_dim - 2))
        self.act_dim = int(d["act_dim"])
        hidden = tuple(d.get("hidden", (128, 128)))
        self.net = ActorCritic(self.obs_dim, self.act_dim, hidden)
        self.net.load_state_dict(d["state_dict"])
        self.net.eval()
        self.norm = RunningNorm(self.sdim)
        self.norm.mean = np.asarray(d.get("norm_mean", np.zeros(self.sdim)), dtype=float)
        self.norm.var = np.asarray(d.get("norm_var", np.ones(self.sdim)), dtype=float)
        self.norm.count = float(d.get("norm_count", 1e-4))

    def _load_ga(self):
        from .brain import MLP
        from .evolution import GA

        ck = GA.load_champion(self.path)
        self.meta.update({k: v for k, v in ck.items() if k != "genome"})
        self.meta["genome_size"] = int(np.asarray(ck["genome"]).size)
        self.mode = "race"
        self.car = str(ck.get("car", "supra"))
        self.sdim = int(ck["obs_size"])
        self.obs_dim = self.sdim
        self.act_dim = int(ck.get("n_actions", 3))
        self.brain = MLP.from_genome(ck["genome"], self.sdim,
                                     tuple(ck["hidden"]), self.act_dim)

    def _adapt_ppo_obs(self, env_obs: np.ndarray) -> np.ndarray:
        env_obs = np.asarray(env_obs, dtype=np.float32).ravel()
        sensor = env_obs[:min(self.sdim, env_obs.size)]
        if sensor.size < self.sdim:
            sensor = np.concatenate([sensor, np.zeros(self.sdim - sensor.size, dtype=np.float32)])
        full = np.concatenate([sensor, _mode_vec(self.mode)]).astype(np.float32)
        if full.size < self.obs_dim:
            full = np.concatenate([full, np.zeros(self.obs_dim - full.size, dtype=np.float32)])
        elif full.size > self.obs_dim:
            full = full[:self.obs_dim]
        snorm = self.norm.normalize(full[:self.sdim])
        return np.concatenate([snorm, full[self.sdim:]]).astype(np.float32)

    def action(self, env_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.kind == "ppo":
            import torch
            nobs = self._adapt_ppo_obs(env_obs)
            with torch.no_grad():
                raw = self.net.act_mean(
                    torch.as_tensor(nobs).unsqueeze(0)
                ).squeeze(0).numpy()
            return raw.astype(np.float32), raw.astype(np.float32), {"nobs": nobs}

        obs = np.asarray(env_obs, dtype=np.float32).ravel()
        sensor = obs[:min(self.sdim, obs.size)]
        if sensor.size < self.sdim:
            sensor = np.concatenate([sensor, np.zeros(self.sdim - sensor.size, dtype=np.float32)])
        raw = np.asarray(self.brain.forward(sensor), dtype=np.float32)
        steer = float(np.clip(raw[0], -1.0, 1.0))
        throttle = float(np.clip(raw[1], 0.0, 1.0)) if raw.size > 1 else 0.0
        brake = float(np.clip(raw[2], 0.0, 1.0)) if raw.size > 2 else 0.0
        env_action = np.array([steer, throttle - brake], dtype=np.float32)
        return raw, env_action, {"nobs": sensor}

    def action_payload(self, raw: np.ndarray, env_action: np.ndarray) -> dict:
        raw = np.asarray(raw, dtype=float)
        if self.kind == "ga":
            throttle = float(np.clip(raw[1], 0.0, 1.0)) if raw.size > 1 else 0.0
            brake = float(np.clip(raw[2], 0.0, 1.0)) if raw.size > 2 else 0.0
            return {"raw": raw.tolist(), "steer": float(raw[0]),
                    "throttle": throttle, "brake": brake, "handbrake": 0.0}
        clipped = np.clip(np.asarray(env_action, dtype=float), -1.0, 1.0)
        long = float(clipped[1]) if len(clipped) > 1 else 0.0
        hb = float(np.clip((clipped[2] + 1.0) / 2.0, 0.0, 1.0)) if len(clipped) > 2 else 0.0
        return {"raw": raw.tolist(), "steer": float(clipped[0]),
                "throttle": max(long, 0.0), "brake": max(-long, 0.0),
                "handbrake": hb}


def _track_payload(trk) -> dict:
    return {
        "length": float(trk.length),
        "width": float(trk.width),
        "elev_gain": float(getattr(trk, "elev_gain", 0.0)),
        "metadata": getattr(trk, "metadata", {}),
        "landmarks": getattr(trk, "landmarks", []),
        "sectors": getattr(trk, "sectors", []),
        "source_confidence": getattr(trk, "source_confidence", None),
        "center": np.asarray(trk.center).tolist(),
        "left": np.asarray(trk.left).tolist(),
        "right": np.asarray(trk.right).tolist(),
        "z": np.asarray(getattr(trk, "z", np.zeros(len(trk.center)))).tolist(),
        "grade": np.asarray(getattr(trk, "grade", np.zeros(len(trk.center)))).tolist(),
        "vcurv": np.asarray(getattr(trk, "vcurv", np.zeros(len(trk.center)))).tolist(),
        "bank": np.asarray(getattr(trk, "bank", np.zeros(len(trk.center)))).tolist(),
        "curvature": np.asarray(trk.curvature).tolist(),
    }


def _natural_speed(trk, idx: int) -> float:
    k = abs(float(trk.curvature[idx]))
    return (min(34.0, (11.0 / k) ** 0.5) * 0.85) if k > 1e-4 else 30.0


def _diagnostic_start_speed(policy: DiagnosticPolicy, trk, idx: int) -> float:
    if idx == 0:
        return 0.0
    if policy.meta.get("ring_pipeline"):
        try:
            from .ring_race import _stage_defaults, natural_speed
            spec = _stage_defaults(policy.meta.get("ring_stage", "survive"))
            return natural_speed(trk, idx, spec)
        except Exception:
            pass
    return _natural_speed(trk, idx)


def _start_indices(trk, starts: list[str]) -> list[int]:
    names = {"line": 0.0, "quarter": 0.25, "half": 0.5, "threequarter": 0.75}
    out = []
    for s in starts:
        s = str(s).strip().lower()
        frac = names.get(s)
        if frac is None:
            try:
                frac = float(s)
            except Exception:
                frac = 0.0
        arc = float(np.clip(frac, 0.0, 0.999)) * float(trk.length)
        out.append(int(np.searchsorted(trk.arc, arc) % len(trk.center)))
    return out or [0]


def _vehicle_payload(veh: Vehicle) -> dict:
    return {
        "x": float(veh.x), "y": float(veh.y), "z": float(veh.z),
        "speed": float(veh.speed), "speed_kmh": float(veh.speed * 3.6),
        "yaw": float(veh.yaw), "yaw_rate": float(veh.r),
        "slip_deg": float(np.degrees(veh.slip_angle)),
        "rpm": float(veh.rpm), "gear": int(veh.gear),
        "airborne": bool(veh.airborne), "air_time": float(veh.air_time),
        "landing_g": float(veh.landing_g),
        "damage": {
            "engine": float(getattr(veh, "engine_damage", 0.0)),
            "aero": float(getattr(veh, "aero_damage", 0.0)),
            "steer_bias": float(getattr(veh, "steer_bias", 0.0)),
            "turbo_broken": bool(getattr(veh, "turbo_broken", False)),
            "drivetrain_broken": bool(getattr(veh, "drivetrain_broken", False)),
            "wheel": np.asarray(getattr(veh, "wheel_damage", np.zeros(4))).tolist(),
        },
    }


def _track_frame_payload(fr: dict) -> dict:
    keys = ("arc", "progress", "lateral", "curvature", "grade", "vcurv",
            "bank", "z", "heading", "off_track", "half_width", "width")
    return {k: (bool(fr[k]) if k == "off_track" else float(fr[k]))
            for k in keys if k in fr}


def _opponents_payload(vehs: list[Vehicle], agent_id: int) -> list[dict]:
    out = []
    ego = vehs[agent_id]
    for j, v in enumerate(vehs):
        if j == agent_id:
            continue
        out.append({
            "agent_id": j,
            "dx": float(v.x - ego.x),
            "dy": float(v.y - ego.y),
            "dist": float(np.hypot(v.x - ego.x, v.y - ego.y)),
            "speed": float(v.speed),
            "damage": {
                "engine": float(getattr(v, "engine_damage", 0.0)),
                "aero": float(getattr(v, "aero_damage", 0.0)),
            },
        })
    return out


def _choose_scenario(policy: DiagnosticPolicy, scenario: str, opponents: list[str]) -> str:
    if opponents:
        return "frozen_opponents"
    if scenario != "auto":
        return scenario
    if policy.kind == "ppo" and policy.obs_dim == 78:
        return "true_multi"
    return "solo"


def _make_env(policy: DiagnosticPolicy, trk, scenario: str, opponents: list[str], seed: int):
    from .ppo_env import SupraEnv
    from .race_env import RaceEnv
    from .multi_env import MultiAgentRaceEnv

    cfg = PPOSpec()
    cfg.random_start = False
    if policy.meta.get("sensor_lookahead_distances"):
        cfg.sensor_lookahead_distances = tuple(policy.meta["sensor_lookahead_distances"])
    if policy.meta.get("sensor_pace_block"):
        cfg.sensor_pace_block = True                     # fable-v1 obs layout
        if policy.meta.get("sensor_pace_distances"):
            cfg.sensor_pace_distances = tuple(policy.meta["sensor_pace_distances"])
    if getattr(trk, "metadata", {}).get("long_track"):
        cfg.episode_seconds = max(cfg.episode_seconds, 540.0)
        cfg.track_profile = trk.metadata.get("profile", "long-track")
    kw = dict(car=policy.car, ppo=cfg, fixed_track=trk, rng_seed=seed,
              diagnostics=True)
    if scenario == "true_multi":
        env = MultiAgentRaceEnv(n_agents=4, **kw)
    elif scenario == "frozen_opponents":
        opp_cfg = [{"checkpoint": p, "car": policy.car} for p in opponents]
        env = RaceEnv(opponent_configs=opp_cfg, **kw)
    elif policy.meta.get("fable_pipeline"):
        from .fable5 import FableEnv, attach_envelope, stage_defaults
        attach_envelope(trk, policy.car)
        cfg.episode_seconds = max(cfg.episode_seconds, 1000.0)
        env = FableEnv(mode=policy.mode,
                       fable_spec=stage_defaults(
                           policy.meta.get("fable_stage", "foundation")),
                       **kw)
    elif policy.meta.get("ring_pipeline"):
        from .ring_race import RingRaceEnv, _stage_defaults
        env = RingRaceEnv(mode=policy.mode,
                          ring_spec=_stage_defaults(policy.meta.get("ring_stage", "survive")),
                          **kw)
    else:
        env = SupraEnv(mode=policy.mode, **kw)
    return env


def _records_to_npz(records: list[dict], path: Path):
    def arr(path_keys, default=0.0):
        vals = []
        for r in records:
            cur = r
            for k in path_keys:
                cur = cur.get(k, {}) if isinstance(cur, dict) else {}
            vals.append(cur if not isinstance(cur, dict) else default)
        return np.asarray(vals)

    obs = [r["observation"]["vector"] for r in records]
    same_obs = len({len(o) for o in obs}) == 1
    np.savez_compressed(
        path,
        episode=arr(["episode"]).astype(np.int32),
        step=arr(["step"]).astype(np.int32),
        agent_id=arr(["agent_id"]).astype(np.int32),
        t=arr(["t"]).astype(np.float32),
        speed=arr(["vehicle", "speed"]).astype(np.float32),
        lateral=arr(["track", "lateral"]).astype(np.float32),
        slip_deg=arr(["vehicle", "slip_deg"]).astype(np.float32),
        yaw_rate=arr(["vehicle", "yaw_rate"]).astype(np.float32),
        reward=arr(["reward", "total"]).astype(np.float32),
        steer=arr(["action", "steer"]).astype(np.float32),
        throttle=arr(["action", "throttle"]).astype(np.float32),
        brake=arr(["action", "brake"]).astype(np.float32),
        progress=arr(["track", "progress"]).astype(np.float32),
        curvature=arr(["track", "curvature"]).astype(np.float32),
        off_track=arr(["track", "off_track"]).astype(bool),
        obs=np.asarray(obs, dtype=np.float32) if same_obs else np.asarray([], dtype=np.float32),
    )


def _detect_events(records: list[dict], track_half: float) -> list[dict]:
    events = []
    recent: dict[tuple, float] = {}
    by_agent: dict[tuple, list[dict]] = {}
    for r in records:
        by_agent.setdefault((r["episode"], r["agent_id"]), []).append(r)

    def add(r, kind, why, severity):
        key = (r["episode"], r["agent_id"], kind)
        if r["t"] - recent.get(key, -1e9) < 1.0:
            return
        recent[key] = r["t"]
        events.append({
            "episode": r["episode"], "agent_id": r["agent_id"],
            "time": r["t"], "step": r["step"], "kind": kind,
            "severity": float(np.clip(severity, 0.0, 1.0)), "why": why,
        })

    for key, rows in by_agent.items():
        steer_hist = []
        actions = []
        obs_head = []
        for i, r in enumerate(rows):
            tr, veh, act = r["track"], r["vehicle"], r["action"]
            parts = r["reward"].get("parts", {})
            reward = float(r["reward"]["total"])
            lat_abs = abs(float(tr["lateral"]))
            speed = float(veh["speed"])
            slip = abs(float(veh["slip_deg"]))
            yaw_rate = abs(float(veh["yaw_rate"]))
            steer = float(act["steer"])
            throttle = float(act["throttle"])
            brake = float(act["brake"])
            if throttle > 0.55 and lat_abs > track_half * 0.9 and reward > 0:
                add(r, "offtrack_farming",
                    "throttle/reward while the car is on or beyond the outer edge",
                    min(1.0, lat_abs / max(track_half, 1e-6)))
            if speed > 22.0 and abs(float(tr["curvature"])) > 0.015 and brake < 0.10:
                add(r, "late_braking",
                    "high speed into a tight section with little brake",
                    min(1.0, speed / 42.0))
            if speed > 5.0 and slip > 35.0 and yaw_rate > 0.75:
                add(r, "oversteer_spin",
                    "large slip angle and yaw rate indicate loss of rotation control",
                    min(1.0, slip / 95.0))
            if i > 0:
                prev_lat = abs(float(rows[i - 1]["track"]["lateral"]))
                if abs(steer) > 0.75 and yaw_rate < 0.25 and lat_abs > prev_lat + 0.15 and speed > 12.0:
                    add(r, "understeer",
                        "high steering demand but lateral error is still growing",
                        min(1.0, (lat_abs - prev_lat) / 1.5))
            if float(parts.get("collision", 0.0)) < 0.0:
                add(r, "collision_failure", "collision or mechanical damage penalty", 1.0)
            if bool(veh["airborne"]) or float(veh["landing_g"]) > 2.5:
                add(r, "jump_crest_failure",
                    "airtime or hard landing occurred during the run",
                    min(1.0, float(veh["landing_g"]) / 5.0 if veh["landing_g"] else 0.6))
            steer_hist.append(np.sign(steer) if abs(steer) > 0.25 else 0.0)
            if len(steer_hist) > 8:
                steer_hist.pop(0)
            nonzero = [s for s in steer_hist if s != 0.0]
            flips = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)
            if flips >= 4 and abs(float(r["track"]["lateral"])) > 1.0:
                add(r, "steering_oscillation",
                    "rapid steering sign changes while line error remains non-trivial",
                    min(1.0, flips / 6.0))
            actions.append([steer, throttle, brake])
            obs_head.append(r["observation"]["vector"][:min(20, len(r["observation"]["vector"]))])
        if len(actions) > 20:
            astd = float(np.mean(np.std(np.asarray(actions), axis=0)))
            ostd = float(np.mean(np.std(np.asarray(obs_head), axis=0))) if obs_head else 0.0
            if astd < 0.025 and ostd > 0.08:
                r0 = rows[0]
                add(r0, "policy_collapse",
                    "actions are nearly constant while observations vary",
                    min(1.0, ostd / 0.25))
    return events


def _summarize(records: list[dict], events: list[dict]) -> dict:
    by_agent: dict[tuple, list[dict]] = {}
    for r in records:
        by_agent.setdefault((r["episode"], r["agent_id"]), []).append(r)
    runs = []
    for (ep, aid), rows in sorted(by_agent.items()):
        reasons = [r.get("termination_reason") for r in rows if r.get("termination_reason")]
        lap_times, invalid_laps = _lap_times(rows)
        dottinger = _max_speed_in_landmark(rows, "Döttinger Höhe")
        runs.append({
            "episode": ep,
            "agent_id": aid,
            "steps": len(rows),
            "max_lap": float(max(r["track"]["progress"] for r in rows)) if rows else 0.0,
            "max_cum_laps": float(max(r.get("info", {}).get("laps", 0.0) for r in rows)) if rows else 0.0,
            "best_clean_lap_time": min(lap_times) if lap_times else None,
            "clean_lap_times": lap_times,
            "invalid_laps": invalid_laps,
            "mean_speed": float(np.mean([r["vehicle"]["speed"] for r in rows])) if rows else 0.0,
            "max_dottinger_speed": dottinger,
            "offtrack_steps": int(sum(1 for r in rows if r["track"]["off_track"])),
            "collision_steps": int(sum(1 for r in rows if r["reward"].get("parts", {}).get("collision", 0.0) < 0.0)),
            "max_slip_deg": float(max(abs(r["vehicle"]["slip_deg"]) for r in rows)) if rows else 0.0,
            "max_landing_g": float(max(r["vehicle"]["landing_g"] for r in rows)) if rows else 0.0,
            "termination_reason": reasons[-1] if reasons else None,
        })
    return {
        "rows": len(records),
        "runs": runs,
        "event_counts": {k: sum(1 for e in events if e["kind"] == k)
                         for k in sorted({e["kind"] for e in events})},
        "mean_speed": float(np.mean([r["vehicle"]["speed"] for r in records])) if records else 0.0,
        "max_speed": float(max((r["vehicle"]["speed"] for r in records), default=0.0)),
        "best_clean_lap_time": min(
            (t for run in runs for t in run["clean_lap_times"]), default=None
        ),
        "invalid_lap_count": int(sum(run["invalid_laps"] for run in runs)),
        "max_dottinger_speed": max(
            (run["max_dottinger_speed"] or 0.0 for run in runs), default=0.0
        ),
        "offtrack_steps": int(sum(1 for r in records if r["track"]["off_track"])),
        "collision_steps": int(sum(1 for r in records if r["reward"].get("parts", {}).get("collision", 0.0) < 0.0)),
        "ring": _ring_summary(records),
    }


def _ring_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    ring_rows = [r for r in records if any(str(k).startswith("ring_") for k in r.get("info", {}))]
    if not ring_rows:
        return {}
    reasons: dict[str, int] = {}
    for r in ring_rows:
        reason = r.get("termination_reason") or r.get("info", {}).get("termination_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    progress = [float(r.get("info", {}).get("ring_progress_m", 0.0)) for r in ring_rows]
    off = [float(r.get("info", {}).get("ring_offtrack_seconds", 0.0)) for r in ring_rows]
    invalid = sum(1 for r in ring_rows if r.get("info", {}).get("ring_invalid"))
    stage = next((r.get("info", {}).get("ring_stage") for r in ring_rows
                  if r.get("info", {}).get("ring_stage")), None)
    max_progress = float(max(progress, default=0.0))
    if reasons:
        rec = "diagnose failure"
    elif stage == "survive":
        rec = "move to fast" if max_progress >= 1000.0 else "keep training survive"
    elif stage == "fast":
        rec = "move to attack" if max_progress >= 0.95 * 20832.0 else "keep training fast"
    else:
        rec = "keep training attack"
    return {
        "stage": stage,
        "max_clean_progress_m": max_progress,
        "offtrack_seconds": float(max(off, default=0.0)),
        "invalid_steps": int(invalid),
        "termination_reason_counts": reasons,
        "recommendation": rec,
    }


def _lap_times(rows: list[dict]) -> tuple[list[float], int]:
    lap_times = []
    invalid_laps = 0
    if not rows:
        return lap_times, invalid_laps
    lap_start_t = float(rows[0]["t"])
    lap_no = 0
    invalid = False
    for r in rows:
        invalid = invalid or bool(r["track"].get("off_track", False))
        laps = float(r.get("info", {}).get("laps", 0.0))
        now_no = int(np.floor(max(0.0, laps)))
        if now_no > lap_no:
            elapsed = float(r["t"]) - lap_start_t
            if invalid:
                invalid_laps += now_no - lap_no
            else:
                lap_times.append(elapsed)
            lap_start_t = float(r["t"])
            invalid = False
            lap_no = now_no
    return lap_times, invalid_laps


def _max_speed_in_landmark(rows: list[dict], name: str) -> float | None:
    if not rows:
        return None
    landmarks = rows[0].get("track_metadata", {}).get("landmarks", [])
    # Rows currently do not duplicate all metadata; fall back to known arc band from
    # track.json's generated Nordschleife asset when the landmark is present there.
    band = None
    for lm in landmarks:
        if lm.get("name") == name:
            band = (float(lm["start_arc"]), float(lm["end_arc"]))
            break
    if band is None:
        # Generated Nordschleife asset places Doettinger Hoehe at the final straight.
        band = (20083.0, 20832.0)
    vals = [r["vehicle"]["speed"] for r in rows
            if band[0] <= float(r["track"].get("arc", 0.0)) <= band[1]]
    return float(max(vals)) if vals else None


def _write_markdown(out_dir: Path, manifest: dict, summary: dict, events: list[dict]):
    lines = [
        "# Agent Diagnostic Telemetry Bundle",
        "",
        "This folder contains raw deterministic replay telemetry. No AI analysis was run.",
        "",
        "## Files",
        "- `raw_trace.jsonl.gz`: one JSON object per control step per agent "
        "(gzip; `zcat` it). Machine consumers should use the .npz instead.",
        "- `raw_trace.npz`: numeric arrays for plotting or scripts.",
        "- `track.json`: track geometry/elevation used for replay.",
        "- `events.json`: rule-based event labels from telemetry.",
        "- `summary.json`: aggregate run stats.",
        "- `plots/`: convenience charts.",
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, default=_jsonable),
        "```",
        "",
        "## Summary",
        f"- Rows: {summary['rows']}",
        f"- Mean speed: {summary['mean_speed']:.2f} m/s",
        f"- Off-track steps: {summary['offtrack_steps']}",
        f"- Collision steps: {summary['collision_steps']}",
        "",
        "## Top Events",
    ]
    for e in events[:30]:
        lines.append(f"- t={e['time']:.2f}s ep={e['episode']} agent={e['agent_id']} "
                     f"{e['kind']} severity={e['severity']:.2f}: {e['why']}")
    (out_dir / "README_FOR_AI.md").write_text("\n".join(lines) + "\n")

    sm = ["# Diagnostic Summary", ""]
    for run in summary["runs"]:
        sm.append(f"- episode {run['episode']} agent {run['agent_id']}: "
                  f"steps {run['steps']}, max laps {run['max_cum_laps']:.3f}, "
                  f"mean speed {run['mean_speed']:.2f} m/s, offtrack {run['offtrack_steps']}, "
                  f"reason {run['termination_reason'] or 'none'}")
    (out_dir / "summary.md").write_text("\n".join(sm) + "\n")


def _plot(records: list[dict], out_dir: Path):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    if not records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (plot_dir / "plot_error.txt").write_text(str(e))
        return

    t = np.asarray([r["t"] for r in records], dtype=float)
    speed = np.asarray([r["vehicle"]["speed"] for r in records], dtype=float)
    lateral = np.asarray([r["track"]["lateral"] for r in records], dtype=float)
    slip = np.asarray([r["vehicle"]["slip_deg"] for r in records], dtype=float)
    steer = np.asarray([r["action"]["steer"] for r in records], dtype=float)
    throttle = np.asarray([r["action"]["throttle"] for r in records], dtype=float)
    brake = np.asarray([r["action"]["brake"] for r in records], dtype=float)
    reward = np.asarray([r["reward"]["total"] for r in records], dtype=float)
    x = np.asarray([r["vehicle"]["x"] for r in records], dtype=float)
    y = np.asarray([r["vehicle"]["y"] for r in records], dtype=float)

    def savefig(name):
        plt.tight_layout()
        plt.savefig(plot_dir / name, dpi=140)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, speed, label="speed m/s")
    plt.plot(t, lateral, label="lateral m")
    plt.plot(t, slip, label="slip deg")
    plt.legend(); plt.xlabel("time s")
    savefig("speed_lateral_slip.png")

    plt.figure(figsize=(10, 5))
    plt.plot(t, steer, label="steer")
    plt.plot(t, throttle, label="throttle")
    plt.plot(t, brake, label="brake")
    plt.legend(); plt.xlabel("time s")
    savefig("controls.png")

    plt.figure(figsize=(10, 5))
    plt.plot(t, reward, label="total reward")
    plt.legend(); plt.xlabel("time s")
    savefig("reward_parts.png")

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, ".", markersize=1.5)
    plt.axis("equal"); plt.xlabel("x"); plt.ylabel("y")
    savefig("track_position.png")


def run_diagnostics(checkpoint: str, track_name: str = "club", scenario: str = "auto",
                    episodes: int = 4, starts: list[str] | None = None,
                    max_steps: int | None = None, out: str | None = None,
                    seed: int = 7, flat: bool = True, hills: bool = False,
                    hill_scale: float | None = None,
                    opponents: list[str] | None = None,
                    raw_trace: str = "gzip") -> Path:
    from .track import configure_hills

    policy = DiagnosticPolicy(checkpoint)
    opponents = opponents or []
    chosen = _choose_scenario(policy, scenario, opponents)
    real_elevation_track = (track_name or "").strip().lower() in (
        "nordschleife", "nurburgring", "nuerburgring"
    )
    effective_flat = bool(policy.legacy or flat or (not real_elevation_track and not hills))
    configure_hills(enabled=bool(hills and not effective_flat),
                    scale=1.0 if hill_scale is None else float(hill_scale),
                    force_flat=effective_flat)
    trk = resolve_track(track_name, seed=seed)
    env = _make_env(policy, trk, chosen, opponents, seed)
    starts = starts or ["line", "quarter", "half", "threequarter"]
    indices = _start_indices(trk, starts)
    steps_limit = int(max_steps or round(env.ppo.episode_seconds * env.ppo.control_hz))
    out_dir = Path(out or Path("diagnostics") / f"{Path(checkpoint).stem}_{track_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # never crash on a name collision (e.g. --out pointing at an existing FILE
    # such as a checkpoint): fall back to a unique diagnostics/ folder
    if out_dir.exists() and not out_dir.is_dir():
        out_dir = Path("diagnostics") / (
            f"{out_dir.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"[diagnose] output path was a file — writing to {out_dir} instead",
              flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for ep in range(int(episodes)):
        idx = indices[ep % len(indices)]
        speed = _diagnostic_start_speed(policy, trk, idx)
        obs = env.reset_at(idx, speed=speed) if hasattr(env, "reset_at") else env.reset()
        for step in range(steps_limit):
            if chosen == "true_multi":
                raw_actions, env_actions, nobs = [], [], []
                for row in obs:
                    raw, act, extra = policy.action(row)
                    raw_actions.append(raw); env_actions.append(act); nobs.append(extra["nobs"])
                next_obs, rewards, terms, truncs, infos = env.step(np.asarray(env_actions, dtype=np.float32))
                vehs = env.vehs
                for aid, veh in enumerate(vehs):
                    fr = trk.frame(veh.x, veh.y)
                    records.append(_row(policy, ep, step, aid, chosen, trk, fr, veh,
                                        raw_actions[aid], env_actions[aid], rewards[aid],
                                        obs[aid], nobs[aid], infos[aid], vehs))
                obs = next_obs
                if bool(np.any(terms)) or bool(np.any(truncs)):
                    break
            else:
                raw, act, extra = policy.action(obs)
                next_obs, reward, term, trunc, info = env.step(act)
                vehs = getattr(env, "_all_vehicles", [env.veh])
                fr = trk.frame(env.veh.x, env.veh.y)
                records.append(_row(policy, ep, step, 0, chosen, trk, fr, env.veh,
                                    raw, act, reward, obs, extra["nobs"], info, vehs))
                obs = next_obs
                if term or trunc:
                    break

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_meta": policy.meta,
        "policy_kind": policy.kind,
        "mode": policy.mode,
        "car": policy.car,
        "track": track_name,
        "scenario": chosen,
        "episodes": int(episodes),
        "starts": starts,
        "max_steps": steps_limit,
        "flat": effective_flat,
        "hill_scale": 1.0 if hill_scale is None else float(hill_scale),
        "opponents": opponents,
    }

    events = _detect_events(records, trk.half)
    summary = _summarize(records, events)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_jsonable))
    (out_dir / "track.json").write_text(json.dumps(_track_payload(trk), default=_jsonable))
    (out_dir / "events.json").write_text(json.dumps(events, indent=2, default=_jsonable))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    # The per-step JSONL trace is human-inspectable but HUGE (200-270 MB/run)
    # and nothing reads it programmatically — the .npz below is the machine
    # form. Default to gzip (~15-20x smaller); "plain" restores the old file,
    # "none" skips it entirely. The .npz is always written.
    mode = (raw_trace or "gzip").lower()
    if mode == "plain":
        with (out_dir / "raw_trace.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r, default=_jsonable) + "\n")
    elif mode != "none":  # "gzip" (default) or any unknown value
        with gzip.open(out_dir / "raw_trace.jsonl.gz", "wt") as f:
            for r in records:
                f.write(json.dumps(r, default=_jsonable) + "\n")
    _records_to_npz(records, out_dir / "raw_trace.npz")
    _write_markdown(out_dir, manifest, summary, events)
    _plot(records, out_dir)
    return out_dir


def _row(policy: DiagnosticPolicy, episode: int, step: int, agent_id: int,
         scenario: str, trk, fr: dict, veh: Vehicle, raw_action, env_action,
         reward: float, obs, nobs, info: dict, vehicles: list[Vehicle]) -> dict:
    return {
        "episode": int(episode),
        "step": int(step),
        "agent_id": int(agent_id),
        "t": float(info.get("t", 0.0)),
        "checkpoint": policy.name,
        "scenario": scenario,
        "track": _track_frame_payload(fr),
        "vehicle": _vehicle_payload(veh),
        "action": policy.action_payload(raw_action, env_action),
        "reward": {
            "total": float(reward),
            "parts": {k: float(v) for k, v in info.get("reward_parts", {}).items()},
        },
        "observation": {
            "sdim": int(policy.sdim),
            "obs_dim": int(policy.obs_dim),
            "env_obs_dim": int(len(np.asarray(obs).ravel())),
            "vector": np.asarray(obs, dtype=float).ravel().tolist(),
            "policy_input": np.asarray(nobs, dtype=float).ravel().tolist(),
        },
        "opponents": _opponents_payload(vehicles, agent_id) if len(vehicles) > 1 else [],
        "info": {k: v for k, v in info.items()
                 if k not in ("reward_parts", "diagnostics")},
        "diagnostics": info.get("diagnostics", {}),
        "termination_reason": info.get("termination_reason"),
    }
