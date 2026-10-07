"""Dedicated Nuerburgring Ring Race training pipeline.

This module is intentionally separate from the generic PPO race path. It reuses
the policy optimiser, vehicle physics, sensors, and 2D viewer, but supplies a
Ring-specific environment, reward, evaluator, checkpoint names, and manifest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import json
import os
import shutil
from typing import Any

import numpy as np
import torch

from .config import PPOSpec, SensorSpec, SimSpec
from .physics import Controls
from .ppo import PPO
from .ppo_env import SupraEnv, V_REF, on_track_factor
from .track import named_track


RING_TRACK = "nordschleife"
RING_CAR = "mazda787b"
RING_PROFILE = "nordschleife-full-20.832km"
RING_BEST = "ring_787b_best.pt"
PIPELINE_MANIFEST = "ring_787b_pipeline.json"
LATEST_EVAL_JSON = "ring_787b_eval_latest.json"
RING_REWARD_VERSION = "ring-race-v2-stability"
LOOKAHEAD = (15.0, 30.0, 55.0, 85.0, 125.0, 180.0)
SPEED_CAP_PROBES = (8.0, 15.0, 22.0, 30.0, 45.0, 65.0, 85.0, 125.0, 180.0)
STAGES = ("survive", "fast", "attack")


@dataclass
class RingRaceReward:
    progress_per_m: float = 0.018
    align: float = 0.045
    safe_speed: float = 0.018
    brake_setup: float = 0.020
    pace: float = 0.0
    straight_attack: float = 0.0
    lap_bonus: float = 25.0
    clean_sector_bonus: float = 0.35
    edge_band: float = 1.8
    offtrack: float = 0.35
    overspeed: float = 0.030
    late_brake: float = 0.030
    corner_slip: float = 0.020
    throttle_slip: float = 0.018
    understeer: float = 0.015
    spin: float = 0.035
    backwards: float = 0.030
    smooth: float = 0.0012
    crash: float = 8.0


@dataclass
class RingRaceSpec:
    stage: str = "survive"
    episode_seconds: float = 180.0
    eval_sector_seconds: float = 60.0
    eval_starts: int = 16
    random_start: bool = True
    critical_start_prob: float = 0.0
    critical_start_fracs: tuple[float, ...] = ()
    critical_eval_seconds: float = 0.0
    lookahead_distances: tuple[float, ...] = LOOKAHEAD
    speed_ref: float = 86.0
    max_speed: float = 90.0
    lat_accel_limit: float = 10.5
    start_speed_scale: float = 0.58
    start_speed_max: float = 26.0
    offtrack_timeout: float = 0.35
    stall_timeout: float = 2.0
    progress_timeout: float = 3.0
    spin_timeout: float = 0.45
    left_track_margin: float = 2.0
    backwards_tol: float = 0.018
    eval_every: int = 25
    reward: RingRaceReward = field(default_factory=RingRaceReward)


def _stage_defaults(stage: str) -> RingRaceSpec:
    stage = _norm_stage(stage)
    if stage == "survive":
        return RingRaceSpec(stage=stage, episode_seconds=180.0,
                            eval_sector_seconds=60.0, eval_every=25,
                            lat_accel_limit=9.0, start_speed_scale=0.52,
                            start_speed_max=24.0,
                            reward=RingRaceReward(
                                progress_per_m=0.020, align=0.050,
                                safe_speed=0.012, brake_setup=0.030,
                                lap_bonus=15.0, offtrack=0.45,
                                overspeed=0.045, late_brake=0.040,
                                crash=10.0))
    if stage == "fast":
        return RingRaceSpec(stage=stage, episode_seconds=360.0,
                            eval_sector_seconds=60.0, eval_every=35,
                            critical_start_prob=0.30,
                            critical_start_fracs=(0.25, 0.50, 0.72, 0.80, 0.90, 0.965),
                            critical_eval_seconds=320.0,
                            lat_accel_limit=10.8, start_speed_scale=0.68,
                            start_speed_max=32.0,
                            reward=RingRaceReward(
                                progress_per_m=0.018, align=0.045,
                                safe_speed=0.020, brake_setup=0.024,
                                pace=0.006, lap_bonus=35.0,
                                overspeed=0.036, late_brake=0.040,
                                crash=9.0))
    return RingRaceSpec(stage=stage, episode_seconds=540.0,
                        eval_sector_seconds=60.0, eval_every=50,
                        critical_start_prob=0.38,
                        critical_start_fracs=(0.25, 0.50, 0.72, 0.80, 0.90, 0.965),
                        critical_eval_seconds=320.0,
                        lat_accel_limit=12.0, start_speed_scale=0.78,
                        start_speed_max=38.0,
                        reward=RingRaceReward(
                            progress_per_m=0.016, align=0.040,
                            safe_speed=0.023, brake_setup=0.018,
                            pace=0.014, straight_attack=0.020,
                            lap_bonus=60.0, overspeed=0.035,
                            late_brake=0.045, crash=10.0))


def _norm_stage(stage: str) -> str:
    stage = (stage or "survive").strip().lower()
    if stage not in (*STAGES, "auto"):
        raise ValueError(f"unknown ring stage '{stage}'")
    return stage


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if hasattr(v, "__dataclass_fields__"):
        return asdict(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _parse_fracs(value) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    out = []
    for part in parts:
        try:
            out.append(float(str(part).strip()))
        except Exception:
            continue
    return tuple(float(np.clip(v, 0.0, 0.999)) for v in out)


def _arc_to_idx(trk, arc: float) -> int:
    return int(np.searchsorted(trk.arc, float(arc) % trk.length) % len(trk.center))


def sector_start_indices(trk, count: int = 16) -> list[int]:
    landmarks = sorted(getattr(trk, "landmarks", []) or [],
                       key=lambda x: float(x.get("start_arc", 0.0)))
    if len(landmarks) >= count:
        picks = np.linspace(0, len(landmarks) - 1, count, dtype=int)
        arcs = [float(landmarks[i].get("start_arc", 0.0)) for i in picks]
    else:
        arcs = [trk.length * i / count for i in range(count)]
    return [_arc_to_idx(trk, a) for a in arcs]


def natural_speed(trk, idx: int, spec: RingRaceSpec) -> float:
    i = int(idx) % len(trk.curvature)
    probes = tuple(sorted(set((*spec.lookahead_distances, *SPEED_CAP_PROBES))))
    look = np.abs(trk.lookahead_curvature(float(trk.arc[i]), probes))
    k = max(abs(float(trk.curvature[i])),
            float(np.max(look)) if len(look) else 0.0)
    if k > 1e-5:
        v = (spec.lat_accel_limit / k) ** 0.5
    else:
        v = spec.max_speed
    return float(np.clip(v * spec.start_speed_scale, 0.0, spec.start_speed_max))


def safe_speed_cap(trk, fr: dict, spec: RingRaceSpec) -> tuple[float, float]:
    probes = tuple(sorted(set((*spec.lookahead_distances, *SPEED_CAP_PROBES))))
    look = np.abs(trk.lookahead_curvature(fr["arc"], probes))
    kmax = max(abs(float(fr["curvature"])), float(np.max(look)) if len(look) else 0.0)
    if kmax <= 1e-5:
        return spec.max_speed, kmax
    cap = (spec.lat_accel_limit / kmax) ** 0.5
    return float(np.clip(cap, 8.0, spec.max_speed)), kmax


class RingRaceEnv(SupraEnv):
    """Solo Ring environment with sector starts and Ring-specific reward."""

    def __init__(self, *args, ring_spec: RingRaceSpec | dict | None = None,
                 **kwargs):
        if ring_spec is None:
            ring_spec = _stage_defaults("survive")
        elif isinstance(ring_spec, dict):
            r = ring_spec.get("reward")
            if isinstance(r, dict):
                ring_spec = dict(ring_spec)
                ring_spec["reward"] = RingRaceReward(**r)
            ring_spec = RingRaceSpec(**ring_spec)
        self.ring_spec = ring_spec
        self.ring_starts: list[int] = []
        self._ring_start_cursor = 0
        kwargs["reward"] = ring_spec.reward
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        if self.trk is not None and not self.ring_starts:
            self.ring_starts = sector_start_indices(self.trk, self.ring_spec.eval_starts)
        if self.trk is not None and self.ring_spec.random_start:
            use_critical = (self.ring_spec.critical_start_fracs
                            and self.rng.random() < float(self.ring_spec.critical_start_prob))
            if use_critical:
                frac = float(self.ring_spec.critical_start_fracs[
                    int(self.rng.integers(len(self.ring_spec.critical_start_fracs)))])
                idx = _arc_to_idx(self.trk, self.trk.length * frac)
            else:
                idx = self.ring_starts[int(self.rng.integers(len(self.ring_starts)))]
            lat = float(self.rng.uniform(-0.18, 0.18)) * self.trk.half
            speed = natural_speed(self.trk, idx, self.ring_spec) * float(self.rng.uniform(0.85, 1.10))
            return self._place_at(idx, speed=speed, lateral=lat,
                                  yaw_jitter=float(self.rng.uniform(-0.08, 0.08)))
        self._reset_ring_bookkeeping()
        return obs

    def reset_at(self, index, speed=0.0):
        if self.trk is None:
            self.trk = self.fixed_track or named_track(RING_TRACK)
        return self._place_at(index, speed=speed)

    def _place_at(self, index: int, speed: float = 0.0, lateral: float = 0.0,
                  yaw_jitter: float = 0.0):
        sx, sy, syaw = self.trk.pose_at(index)
        if lateral:
            nx, ny = self.trk.normal[int(index) % len(self.trk.normal)]
            sx += nx * lateral
            sy += ny * lateral
        self.veh.reset(sx, sy, syaw + yaw_jitter, speed=speed)
        self.box.__init__(self.spec)
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                          fr["z"], fr["vcurv"])
        self.prev_frac = fr["progress"]
        self.cum = 0.0
        self.max_cum = 0.0
        self.best_cum = 0.0
        self.t = 0.0
        self.total_dist = 0.0
        self.off_t = 0.0
        self.stall_t = 0.0
        self.since_prog = 0.0
        self.spin_t = 0.0
        self.step_count = 0
        self.drift_steps = 0
        self.ep_air_time = 0.0
        self.ep_jumps = 0
        self.ep_max_lg = 0.0
        self._was_air = False
        self.prev_action = np.zeros(self.action_dim)
        self._reset_ring_bookkeeping()
        return self._obs()

    def _reset_ring_bookkeeping(self):
        self.ring_invalid = False
        self.ring_offtrack_total = 0.0
        self.ring_terminal_reason = None
        self.ring_progress_m = 0.0
        self._prev_lat_abs = 0.0

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        steer = float(a[0])
        long = float(a[1])
        throttle = max(long, 0.0)
        brake = max(-long, 0.0)

        spec = self.ring_spec
        rw = spec.reward
        dt = self.sim.dt
        reward = 0.0
        reward_parts = {
            "progress_m": 0.0, "align": 0.0, "safe_speed": 0.0,
            "brake_setup": 0.0, "pace": 0.0, "straight_attack": 0.0,
            "offtrack": 0.0, "overspeed": 0.0, "late_brake": 0.0,
            "corner_slip": 0.0, "throttle_slip": 0.0,
            "understeer": 0.0, "spin": 0.0, "backwards": 0.0,
            "lap_bonus": 0.0, "smooth": 0.0, "crash": 0.0,
        }
        diag_values = {"safe_speed_cap": spec.max_speed, "kmax": 0.0,
                       "on_track_factor": 1.0, "heading_error": 0.0,
                       "spin_yaw_limit": 1.15, "path_yaw_rate": 0.0}
        terminated = False
        termination_reason = None
        last_fr = None
        prev_cum = self.cum

        for _ in range(self.control_period):
            fr = self.trk.frame(self.veh.x, self.veh.y)
            last_fr = fr
            self.veh.surface_grip = self.spec.offtrack_grip if fr["off_track"] else 1.0
            self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                              fr["z"], fr["vcurv"])

            frac = fr["progress"]
            d = frac - self.prev_frac
            if d < -0.5:
                d += 1.0
            elif d > 0.5:
                d -= 1.0
            self.prev_frac = frac
            prev_cum = self.cum
            self.cum += d
            self.max_cum = max(self.max_cum, self.cum)
            delta_m = d * self.trk.length
            self.ring_progress_m = max(self.ring_progress_m, self.max_cum * self.trk.length)

            head_err = (self.veh.yaw - fr["heading"] + np.pi) % (2 * np.pi) - np.pi
            ot = on_track_factor(fr, self.trk, rw.edge_band)
            vcap, kmax = safe_speed_cap(self.trk, fr, spec)
            diag_values.update(safe_speed_cap=vcap, kmax=kmax,
                               on_track_factor=ot, heading_error=head_err)

            p_term = rw.progress_per_m * delta_m
            if delta_m < 0.0:
                p_term *= 2.0
            a_term = rw.align * max(0.0, np.cos(head_err)) * (self.veh.speed / V_REF)
            speed_gate = max(0.0, 1.0 - max(0.0, self.veh.speed - vcap) / max(vcap, 1e-6))
            s_term = rw.safe_speed * np.clip(self.veh.speed / max(vcap, 1e-6), 0.0, 1.0) * speed_gate
            step_reward = (p_term + a_term + s_term) * ot
            reward_parts["progress_m"] += p_term * ot
            reward_parts["align"] += a_term * ot
            reward_parts["safe_speed"] += s_term * ot

            if kmax > 0.010 and self.veh.speed > vcap * 0.85 and brake > 0.08:
                b_term = rw.brake_setup * min(1.0, brake) * min(1.4, self.veh.speed / max(vcap, 1e-6))
                step_reward += b_term * ot
                reward_parts["brake_setup"] += b_term * ot
            if rw.pace:
                pace_term = rw.pace * (delta_m / max(dt, 1e-6)) / spec.speed_ref
                step_reward += pace_term * ot
                reward_parts["pace"] += pace_term * ot
            if rw.straight_attack and kmax < 0.004 and abs(head_err) < 0.10:
                st = rw.straight_attack * throttle * np.clip(self.veh.speed / spec.speed_ref, 0.0, 1.15)
                step_reward += st * ot
                reward_parts["straight_attack"] += st * ot

            overspeed = max(0.0, self.veh.speed - vcap)
            if overspeed > 0.5:
                pen = rw.overspeed * (overspeed / max(vcap, 1e-6)) ** 2 * (1.0 + 40.0 * kmax)
                step_reward -= pen
                reward_parts["overspeed"] -= pen
            if kmax > 0.012 and self.veh.speed > vcap * 1.05 and brake < 0.08:
                pen = rw.late_brake * min(2.0, self.veh.speed / max(vcap, 1e-6))
                step_reward -= pen
                reward_parts["late_brake"] -= pen

            if fr["off_track"]:
                self.off_t += dt
                self.ring_offtrack_total += dt
                self.ring_invalid = True
                step_reward -= rw.offtrack
                reward_parts["offtrack"] -= rw.offtrack
            else:
                self.off_t = 0.0

            lat_abs = abs(float(fr["lateral"]))
            yaw_rate = abs(float(self.veh.r))
            slip = abs(float(np.degrees(self.veh.slip_angle)))
            if kmax > 0.012 and slip > 14.0 and self.veh.speed > 10.0:
                slip_x = min(2.0, (slip - 14.0) / 28.0)
                edge_x = max(0.0, lat_abs / max(float(fr.get("half_width", self.trk.half)), 1e-6) - 0.45)
                pen = rw.corner_slip * slip_x * slip_x * (1.0 + edge_x)
                step_reward -= pen
                reward_parts["corner_slip"] -= pen
                if throttle > 0.25:
                    tpen = rw.throttle_slip * slip_x * min(1.4, throttle + 0.25)
                    step_reward -= tpen
                    reward_parts["throttle_slip"] -= tpen
            if abs(steer) > 0.72 and yaw_rate < 0.22 and lat_abs > self._prev_lat_abs + 0.05 and self.veh.speed > 12.0:
                pen = rw.understeer * min(2.0, lat_abs / max(self.trk.half, 1e-6))
                step_reward -= pen
                reward_parts["understeer"] -= pen
            self._prev_lat_abs = lat_abs
            path_yaw_rate = self.veh.speed * max(abs(float(fr["curvature"])), kmax)
            yaw_limit = max(1.15, path_yaw_rate * 1.35 + 0.45)
            spin_slip = slip > 46.0 or (slip > 32.0 and abs(head_err) > 0.35)
            spin_yaw = yaw_rate > yaw_limit and abs(head_err) > 0.35
            spinning = self.veh.speed > 8.0 and (spin_slip or spin_yaw)
            diag_values.update(spin_yaw_limit=yaw_limit,
                               path_yaw_rate=path_yaw_rate)
            self.spin_t = self.spin_t + dt if spinning else max(0.0, self.spin_t - dt)
            if spinning:
                pen = rw.spin * min(2.0, max(slip / 65.0, yaw_rate / max(yaw_limit, 1e-6)))
                step_reward -= pen
                reward_parts["spin"] -= pen

            if int(self.cum) > int(prev_cum) and d > 0:
                step_reward += rw.lap_bonus
                reward_parts["lap_bonus"] += rw.lap_bonus
            crawl = self.t > 1.0 and self.veh.speed < 0.8
            self.stall_t = self.stall_t + dt if crawl else 0.0
            if self.cum > self.best_cum + 0.00025:
                self.best_cum = self.cum
                self.since_prog = 0.0
            else:
                self.since_prog += dt

            if self.off_t > spec.offtrack_timeout:
                terminated = True; termination_reason = "offtrack_timeout"
            elif lat_abs > float(fr.get("half_width", self.trk.half)) + spec.left_track_margin:
                terminated = True; termination_reason = "left_track"
            elif self.stall_t > spec.stall_timeout:
                terminated = True; termination_reason = "stall"
            elif self.since_prog > spec.progress_timeout:
                terminated = True; termination_reason = "no_progress"
            elif self.cum < self.max_cum - spec.backwards_tol:
                pen = rw.backwards
                step_reward -= pen
                reward_parts["backwards"] -= pen
                terminated = True; termination_reason = "backwards"
            elif self.spin_t > spec.spin_timeout:
                terminated = True; termination_reason = "spin"
            elif (getattr(self.veh, "engine_damage", 0.0) > 0.75
                  or getattr(self.veh, "drivetrain_broken", False)):
                terminated = True; termination_reason = "severe_damage"

            reward += step_reward
            clutch, up, down = self.box.update(self.veh, throttle, dt)
            self.veh.step(Controls(steer=steer, throttle=throttle, brake=brake,
                                   clutch=clutch, handbrake=0.0,
                                   shift_up=up, shift_down=down))
            self.t += dt
            self.total_dist += self.veh.speed * dt
            if self.veh.airborne:
                self.ep_air_time += dt
                if not self._was_air:
                    self.ep_jumps += 1
            self._was_air = self.veh.airborne
            self.ep_max_lg = max(self.ep_max_lg, self.veh.landing_g)
            if terminated:
                break

        smooth = -rw.smooth * float(np.sum((a - self.prev_action) ** 2))
        reward += smooth
        reward_parts["smooth"] = smooth
        self.prev_action = a
        if terminated:
            reward -= rw.crash
            reward_parts["crash"] = -rw.crash
            self.ring_terminal_reason = termination_reason

        truncated = (not terminated) and (self.t >= self.ppo.episode_seconds)
        if truncated:
            termination_reason = "time_limit"
        info = {
            "laps": self.max_cum,
            "speed": self.veh.speed,
            "t": self.t,
            "difficulty": self.difficulty,
            "airtime": self.ep_air_time,
            "jumps": self.ep_jumps,
            "max_landing_g": self.ep_max_lg,
            "ring_stage": spec.stage,
            "ring_progress_m": float(max(0.0, self.max_cum * self.trk.length)),
            "ring_clean": bool(not self.ring_invalid and not terminated),
            "ring_invalid": bool(self.ring_invalid),
            "ring_offtrack_seconds": float(self.ring_offtrack_total),
            "ring_safe_speed_cap": float(diag_values["safe_speed_cap"]),
        }
        if self.diagnostics:
            info["reward_parts"] = {k: float(v) for k, v in reward_parts.items()
                                    if abs(float(v)) > 1e-12}
            info["diagnostics"] = {k: float(v) for k, v in diag_values.items()}
            info["termination_reason"] = termination_reason
        return self._obs(), float(reward), terminated, truncated, info


class RingRaceEvaluator:
    def __init__(self, spec: RingRaceSpec, track, manifest_path: str | Path = PIPELINE_MANIFEST):
        self.spec = copy.deepcopy(spec)
        self.track = track
        self.manifest_path = Path(manifest_path)
        self.latest: dict[str, Any] = {}

    def __call__(self, ppo: PPO) -> dict[str, Any]:
        latest = self.evaluate(ppo)
        self.latest = latest
        self._write_manifest(ppo, latest)
        return latest

    def metadata(self) -> dict[str, Any]:
        return {
            "ring_pipeline": True,
            "ring_stage": self.spec.stage,
            "ring_reward_version": RING_REWARD_VERSION,
            "ring_eval": self.latest,
        }

    @torch.no_grad()
    def evaluate(self, ppo: PPO) -> dict[str, Any]:
        cfg = copy.copy(ppo.cfg)
        cfg.random_start = False
        cfg.episode_seconds = max(float(cfg.episode_seconds), self.spec.eval_sector_seconds)
        env = RingRaceEnv(mode="race", car=ppo.car, ppo=cfg, sim=ppo.sim_cfg,
                          fixed_track=self.track, ring_spec=self.spec,
                          rng_seed=880, diagnostics=True)
        starts = sector_start_indices(self.track, self.spec.eval_starts)
        sector_rows = []
        steps_per = int(round(self.spec.eval_sector_seconds * cfg.control_hz))
        for idx in starts:
            sector_rows.append(self._run_from(ppo, env, idx, natural_speed(self.track, idx, self.spec),
                                             steps_per))

        critical_rows = []
        if self.spec.critical_start_fracs and self.spec.critical_eval_seconds > 0:
            critical_steps = int(round(min(float(self.spec.episode_seconds),
                                           float(self.spec.critical_eval_seconds))
                                       * cfg.control_hz))
            for frac in self.spec.critical_start_fracs:
                idx = _arc_to_idx(self.track, self.track.length * float(frac))
                critical_rows.append(
                    self._run_from(ppo, env, idx, natural_speed(self.track, idx, self.spec),
                                   critical_steps)
                )

        line_steps = int(round(self.spec.episode_seconds * cfg.control_hz))
        line = self._run_from(ppo, env, 0, 0.0, line_steps)
        clean_sectors = sum(1 for r in sector_rows if r["clean"])
        clean_chain = self._clean_chain(sector_rows)
        terminal_rate = sum(1 for r in sector_rows if r["termination_reason"] not in (None, "time_limit")) / max(1, len(sector_rows))
        critical_clean = sum(1 for r in critical_rows if r["clean"])
        critical_terminal_rate = sum(
            1 for r in critical_rows
            if r["termination_reason"] not in (None, "time_limit")
        ) / max(1, len(critical_rows))
        clean_progress_m = float(np.mean([r["progress_m"] for r in sector_rows])) if sector_rows else 0.0
        max_clean_progress_m = float(max([line["progress_m"],
                                          *[r["progress_m"] for r in sector_rows],
                                          *[r["progress_m"] for r in critical_rows]],
                                         default=0.0))
        offtrack_seconds = float(sum(r["offtrack_seconds"] for r in sector_rows))
        critical_offtrack_seconds = float(sum(r["offtrack_seconds"] for r in critical_rows))
        best_clean_lap = line["lap_time"] if line["clean_lap"] else None
        invalid_laps = int(0 if line["clean_lap"] else (1 if line["laps"] >= 1.0 else 0))
        max_dottinger = max([line["max_dottinger_speed"],
                             *[r["max_dottinger_speed"] for r in sector_rows],
                             *[r["max_dottinger_speed"] for r in critical_rows]],
                            default=0.0)

        if self.spec.stage == "survive":
            metric = (clean_sectors / max(1, len(sector_rows))) * 10.0
            metric += clean_progress_m / max(1.0, self.track.length)
            metric -= terminal_rate * 5.0 + offtrack_seconds * 0.05
            rec = "move to fast" if clean_sectors >= 14 and terminal_rate <= 0.10 and offtrack_seconds / max(1, len(sector_rows)) <= 0.25 else "keep training survive"
        elif self.spec.stage == "fast":
            metric = clean_chain + max_clean_progress_m / max(1.0, self.track.length) * 4.0
            metric -= critical_terminal_rate * 2.0 + critical_offtrack_seconds * 0.04
            if best_clean_lap:
                metric += 10000.0 / max(best_clean_lap, 1.0)
            rec = "move to attack" if (best_clean_lap or max_clean_progress_m >= self.track.length * 0.95) and critical_terminal_rate <= 0.05 else "keep training fast"
        else:
            metric = (100000.0 / best_clean_lap) if best_clean_lap else max_clean_progress_m / max(1.0, self.track.length) * 12.0
            metric -= invalid_laps * 2.0
            metric -= terminal_rate * 1.5 + critical_terminal_rate * 3.0
            metric -= offtrack_seconds * 0.02 + critical_offtrack_seconds * 0.05
            rec = "keep attacking clean lap time" if best_clean_lap and critical_terminal_rate <= 0.05 else "diagnose failure" if terminal_rate > 0.5 or critical_terminal_rate > 0.25 else "keep training attack"

        termination_counts: dict[str, int] = {}
        for r in [line, *sector_rows, *critical_rows]:
            reason = r["termination_reason"] or "none"
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        latest = {
            "stage": self.spec.stage,
            "metric": float(metric),
            "clean_sectors": int(clean_sectors),
            "sector_count": int(len(sector_rows)),
            "clean_chain": int(clean_chain),
            "terminal_rate": float(terminal_rate),
            "clean_progress_m": clean_progress_m,
            "max_clean_progress_m": max_clean_progress_m,
            "clean_progress_frac": float(max_clean_progress_m / max(1.0, self.track.length)),
            "offtrack_seconds": offtrack_seconds,
            "critical_clean": int(critical_clean),
            "critical_count": int(len(critical_rows)),
            "critical_terminal_rate": float(critical_terminal_rate),
            "critical_offtrack_seconds": critical_offtrack_seconds,
            "best_clean_lap": best_clean_lap,
            "invalid_laps": invalid_laps,
            "mean_speed": float(np.mean([r["mean_speed"] for r in [*sector_rows, *critical_rows]])) if (sector_rows or critical_rows) else 0.0,
            "max_dottinger_speed": float(max_dottinger),
            "termination_counts": termination_counts,
            "recommendation": rec,
            "laps": float(line["laps"]),
            "drift": 0.0,
            "log": self._log_line(metric, clean_sectors, len(sector_rows), clean_chain,
                                  best_clean_lap, invalid_laps, max_dottinger,
                                  terminal_rate, max_clean_progress_m,
                                  offtrack_seconds,
                                  float(np.mean([r["mean_speed"] for r in [*sector_rows, *critical_rows]])) if (sector_rows or critical_rows) else 0.0,
                                  critical_clean, len(critical_rows),
                                  critical_terminal_rate),
        }
        return latest

    def _run_from(self, ppo: PPO, env: RingRaceEnv, idx: int, speed: float,
                  steps: int) -> dict[str, Any]:
        obs = env.reset_at(idx, speed=speed)
        info: dict[str, Any] = {}
        speeds = []
        max_dottinger = 0.0
        lap_time = None
        clean_lap = False
        for _ in range(max(1, steps)):
            nobs = ppo._norm(obs[None])[0]
            a = ppo.net.act_mean(ppo._t(nobs).unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(a)
            speeds.append(float(info.get("speed", 0.0)))
            fr = env.trk.frame(env.veh.x, env.veh.y)
            if 20083.0 <= float(fr.get("arc", 0.0)) <= 20832.0:
                max_dottinger = max(max_dottinger, float(info.get("speed", 0.0)))
            if info.get("laps", 0.0) >= 1.0 and not info.get("ring_invalid"):
                lap_time = float(info.get("t", 0.0))
                clean_lap = True
                break
            if term or trunc:
                break
        reason = info.get("termination_reason")
        clean = bool(reason in (None, "time_limit") and not info.get("ring_invalid", False))
        return {
            "clean": clean,
            "clean_lap": clean_lap,
            "lap_time": lap_time,
            "termination_reason": reason,
            "progress_m": float(info.get("ring_progress_m", 0.0)),
            "laps": float(info.get("laps", 0.0)),
            "offtrack_seconds": float(info.get("ring_offtrack_seconds", 0.0)),
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_dottinger_speed": max_dottinger,
        }

    @staticmethod
    def _clean_chain(rows: list[dict[str, Any]]) -> int:
        best = cur = 0
        for row in rows:
            cur = cur + 1 if row["clean"] else 0
            best = max(best, cur)
        return best

    def _log_line(self, metric, clean_sectors, sector_count, clean_chain,
                  best_clean_lap, invalid_laps, max_dottinger, terminal_rate,
                  max_clean_progress_m, offtrack_seconds, mean_speed,
                  critical_clean=0, critical_count=0,
                  critical_terminal_rate=0.0) -> str:
        lap = 0.0 if best_clean_lap is None else float(best_clean_lap)
        crit = (f" hard={critical_clean}/{critical_count} "
                f"hard_term={critical_terminal_rate:.2f}"
                if critical_count else "")
        if self.spec.stage == "survive":
            return (f"[eval-ring] stage=survive clean_sectors={clean_sectors}/{sector_count} "
                    f"clean_progress={max_clean_progress_m:.0f}m terminal_rate={terminal_rate:.2f} "
                    f"offtrack_seconds={offtrack_seconds:.2f} mean_speed={mean_speed:.2f} "
                    f"clean_chain={clean_chain}{crit} "
                    f"metric={metric:.3f}")
        if self.spec.stage == "fast":
            return (f"[eval-ring] stage=fast clean_chain={clean_chain} "
                    f"clean_sectors={clean_sectors}/{sector_count} clean_lap={lap:.2f} "
                    f"clean_progress={max_clean_progress_m:.0f}m terminal_rate={terminal_rate:.2f} "
                    f"offtrack_seconds={offtrack_seconds:.2f} mean_speed={mean_speed:.2f}{crit} "
                    f"metric={metric:.3f}")
        return (f"[eval-ring] stage=attack best_clean_lap={lap:.2f} "
                f"invalid_laps={invalid_laps} max_dottinger={max_dottinger:.1f} "
                f"clean_progress={max_clean_progress_m:.0f}m terminal_rate={terminal_rate:.2f} "
                f"offtrack_seconds={offtrack_seconds:.2f} mean_speed={mean_speed:.2f}{crit} "
                f"clean_chain={clean_chain} "
                f"metric={metric:.3f}")

    def _write_manifest(self, ppo: PPO, latest: dict[str, Any],
                        checkpoint: str | None = None):
        old = {}
        if self.manifest_path.exists():
            try:
                old = json.loads(self.manifest_path.read_text())
            except Exception:
                old = {}
        stages = dict(old.get("stages") or {})
        stages[self.spec.stage] = latest
        manifest = {
            "pipeline": "ring_race",
            "track": RING_TRACK,
            "car": ppo.car,
            "current_stage": self.spec.stage,
            "active_checkpoint": checkpoint or old.get("active_checkpoint"),
            "best_checkpoint": RING_BEST if Path(RING_BEST).exists() else old.get("best_checkpoint"),
            "latest_eval": latest,
            "stages": stages,
            "stage_gates": {
                "survive_to_fast": "14/16 clean sector starts, terminal_rate <= 0.10, mean offtrack <= 0.25s",
                "fast_to_attack": "clean lap or >= 95% clean progress from line",
                "attack": "optimize fastest clean lap until plateau",
            },
            "latest_diagnostic_folder": old.get("latest_diagnostic_folder"),
            "recommended_next_action": latest.get("recommendation"),
            "reseed_count": int(old.get("reseed_count", 0) or 0),
            "ring_reward_version": RING_REWARD_VERSION,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=_jsonable) + "\n")
        Path(LATEST_EVAL_JSON).write_text(json.dumps(latest, indent=2, default=_jsonable) + "\n")


def checkpoint_for_stage(stage: str, out: str | None = None) -> str:
    if out:
        return out if out.endswith(".pt") else out + ".pt"
    return f"ring_787b_{_norm_stage(stage)}.pt"


def default_resume_for(stage: str) -> str | None:
    if stage == "fast":
        return "ring_787b_survive_best.pt"
    if stage == "attack":
        return "ring_787b_fast_best.pt"
    return None


def _configure_ppo(spec: RingRaceSpec, *, workers=None, pop=None, anneal=False,
                   lr=None, patience=None, max_restarts=None,
                   ent_coef=None, init_log_std=None) -> PPOSpec:
    cfg = PPOSpec()
    cfg.episode_seconds = spec.episode_seconds
    cfg.random_start = True
    cfg.sensor_lookahead_distances = spec.lookahead_distances
    cfg.track_profile = RING_PROFILE
    cfg.n_envs = int(pop or 8)
    cfg.n_workers = int(workers or min(8, cfg.n_envs))
    if cfg.n_workers < 1 or cfg.n_envs % cfg.n_workers != 0:
        cfg.n_workers = 1
    cfg.anneal = bool(anneal or spec.stage in ("fast", "attack"))
    cfg.ent_coef = 0.006 if spec.stage == "survive" else 0.004 if spec.stage == "fast" else 0.0035
    cfg.init_log_std = -0.35 if spec.stage == "survive" else -0.55 if spec.stage == "fast" else -0.55
    if ent_coef is not None:
        cfg.ent_coef = max(0.0, float(ent_coef))
    if init_log_std is not None:
        cfg.init_log_std = float(init_log_std)
    if lr is not None:
        cfg.lr = float(lr)
    elif spec.stage == "attack":
        cfg.lr = 1.5e-4
    if patience is not None:
        cfg.specialist_patience = max(0, int(patience))
    else:
        cfg.specialist_patience = 650 if spec.stage == "survive" else 850 if spec.stage == "fast" else 1200
    if max_restarts is not None:
        cfg.max_restarts = max(0, int(max_restarts))
    else:
        cfg.max_restarts = 4 if spec.stage == "survive" else 6
    return cfg


def _copy_best_if_present(best_path: str, stage: str):
    if os.path.exists(best_path):
        shutil.copy2(best_path, RING_BEST)
        print(f"[ring-race] {stage} best promoted -> {RING_BEST}", flush=True)


def _train_one_stage(stage: str, iterations: int, *, car=RING_CAR, resume=None,
                     out=None, live=False, workers=None, pop=None, anneal=False,
                     lr=None, patience=None, max_restarts=None,
                     critical_starts=None, critical_prob=None, ent_coef=None,
                     init_log_std=None, force_log_std=None,
                     reset_optimizer=False, promote_best=True,
                     open_best_floor=False) -> dict[str, Any]:
    spec = _stage_defaults(stage)
    crit = _parse_fracs(critical_starts)
    if crit:
        spec.critical_start_fracs = crit
        if spec.critical_eval_seconds <= 0:
            spec.critical_eval_seconds = min(320.0, spec.episode_seconds)
    if critical_prob is not None:
        spec.critical_start_prob = float(np.clip(float(critical_prob), 0.0, 1.0))
    trk = named_track(RING_TRACK)
    cfg = _configure_ppo(spec, workers=workers, pop=pop, anneal=anneal,
                         lr=lr, patience=patience, max_restarts=max_restarts,
                         ent_coef=ent_coef, init_log_std=init_log_std)
    ckpt = checkpoint_for_stage(stage, out)
    manifest_path = Path(PIPELINE_MANIFEST)
    evaluator = RingRaceEvaluator(spec, trk, manifest_path)
    resume_path = resume
    if not resume_path:
        dflt = default_resume_for(stage)
        if dflt and os.path.exists(dflt):
            resume_path = dflt

    if live:
        from .app_ppo import run as live_run
        live_run(car=car, iterations=iterations, checkpoint=ckpt, mode="race",
                 resume=resume_path, fixed_track=trk, track_name=RING_TRACK,
                 ppo_cfg=cfg, reward=spec.reward,
                 env_cls_override=RingRaceEnv,
                 env_kwargs={"ring_spec": spec},
                 eval_callback=evaluator,
                 extra_metadata=evaluator.metadata)
        latest = evaluator.latest
    else:
        ppo = PPO(mode="race", car=car, ppo=cfg, reward=spec.reward,
                  fixed_track=trk, track_name=RING_TRACK,
                  env_cls_override=RingRaceEnv,
                  env_kwargs={"ring_spec": spec},
                  eval_callback=evaluator,
                  extra_metadata=evaluator.metadata)
        if resume_path:
            if os.path.exists(resume_path):
                meta = ppo.load_state(resume_path)
                if meta.get("ring_stage") != stage:
                    ppo.resumed_metric = -1e9
                if open_best_floor:
                    ppo.resumed_metric = -1e9
                if force_log_std is not None:
                    with torch.no_grad():
                        ppo.net.log_std.data.fill_(float(force_log_std))
                        ppo.net.log_std.data.clamp_(ppo.net.LOG_STD_MIN,
                                                    ppo.net.LOG_STD_MAX)
                    reset_optimizer = True
                if reset_optimizer:
                    ppo.opt = torch.optim.Adam(ppo.net.parameters(), lr=cfg.lr)
                print(f"[ring-race] resumed {resume_path} -> stage {stage} "
                      f"(updates {ppo.updates})", flush=True)
            else:
                print(f"[ring-race] resume not found: {resume_path}; starting fresh",
                      flush=True)
        print(f"Ring Race [{stage}]: {cfg.n_envs} envs x {cfg.rollout} steps, "
              f"episode cap {cfg.episode_seconds:.0f}s -> {ckpt}", flush=True)
        save_every = max(1, min(spec.eval_every, int(iterations)))
        ppo.train(iterations=iterations, checkpoint=ckpt, log_every=5,
                  save_every=save_every)
        if not evaluator.latest:
            ppo.eval_and_save(ckpt, PPO.best_path_for(ckpt))
        latest = evaluator.latest
    best_path = PPO.best_path_for(ckpt)
    if promote_best:
        _copy_best_if_present(best_path, stage)
    elif os.path.exists(best_path):
        print(f"[ring-race] {stage} best kept local at {best_path} "
              f"(promotion disabled)", flush=True)
    evaluator._write_manifest(type("P", (), {"car": car})(), latest, checkpoint=ckpt)
    return latest


def run_ring_race(iterations: int, stage: str = "survive", **kwargs) -> dict[str, Any]:
    stage = _norm_stage(stage)
    iterations = max(1, int(iterations))
    if stage != "auto":
        return _train_one_stage(stage, iterations, **kwargs)

    splits = [("survive", 0.35), ("fast", 0.35), ("attack", 0.30)]
    latest: dict[str, Any] = {}
    for st, frac in splits:
        iters = max(1, int(round(iterations * frac)))
        stage_kwargs = dict(kwargs)
        stage_kwargs["out"] = None
        stage_kwargs["resume"] = None
        latest = _train_one_stage(st, iters, **stage_kwargs)
        if st == "survive" and latest.get("recommendation") != "move to fast":
            print("[ring-race] auto stopped after survive: gate not met", flush=True)
            break
        if st == "fast" and latest.get("recommendation") != "move to attack":
            print("[ring-race] auto stopped after fast: gate not met", flush=True)
            break
    return latest


def watch_ring_race(checkpoint: str = RING_BEST, car: str | None = None,
                    seed: int = 7, audio_on: bool = True,
                    max_frames: int | None = None):
    from .aiviz import PolicyAgent
    from .viewer2 import run as run_app

    if not os.path.exists(checkpoint):
        raise SystemExit(f"No Ring Race policy at '{checkpoint}'. Train one with --ring-race first.")
    net, norm, meta = PPO.load_policy(checkpoint)
    agent = PolicyAgent(net, norm, meta, mode="race")
    trk = named_track(RING_TRACK)
    sensor_spec = None
    if meta.get("sensor_lookahead_distances"):
        sensor_spec = SensorSpec()
        sensor_spec.lookahead_distances = tuple(meta["sensor_lookahead_distances"])
    run_app(car=car or meta.get("car", RING_CAR), seed=seed, audio_on=audio_on,
            controller=agent.act, track=trk,
            title=f"Ring Race Watch - {os.path.basename(checkpoint)}",
            agent=agent, sensor_spec=sensor_spec,
            max_frames=max_frames)
