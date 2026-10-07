"""Gymnasium environment: score as many trick points as possible in one run."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drift.config import (CURRICULUM, ArenaConfig, CarConfig, RewardConfig, ScoreConfig,
                          Stage)
from drift.physics import CarModel
from drift.scoring import ScoreKeeper
from drift.track import Arena

N_PILLAR_SLOTS = 3


class DriftEnv(gym.Env):
    """Drive a rear-drive car around a walled arena and chain drift tricks.

    The action is what a driver has: steering, a combined throttle/brake pedal
    and the handbrake. The reward is the arcade score itself - points accrue
    while the car is sideways, scale with a combo multiplier that only tricks
    can push to the top, and are largely forfeited on a spin or a crash.

    Observations are deliberately ego-centric and sensor-like: chassis states,
    slip angles, a rangefinder sweep, the nearest pillars in body frame, and
    the live combo state (without which the task is not Markov - the policy
    cannot decide whether to bank or push without knowing what is at stake).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, stage: Stage | None = None, car: CarConfig | None = None,
                 arena: ArenaConfig | None = None, score: ScoreConfig | None = None,
                 reward: RewardConfig | None = None):
        super().__init__()
        self.car_cfg = car or CarConfig()
        self.score_cfg = score or ScoreConfig()
        self.reward_cfg = reward or RewardConfig()
        self.stage = stage if stage is not None else CURRICULUM[-1]

        self.model = CarModel(self.car_cfg)
        self.arena = Arena(arena or ArenaConfig(), self.stage)
        self.keeper = ScoreKeeper(self.score_cfg)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        obs_dim = 9 + 6 + self.arena.cfg.n_beams + 4 + 4 * N_PILLAR_SLOTS
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.state = self.model.spawn(0.0, 0.0, 0.0, 0.0)
        self.step_count = 0
        self.max_steps = 1
        self.stall_t = 0.0
        self._prev_gap = 0.0
        self.trail: list[tuple[float, float, float]] = []
        self.last_events: list = []

    # ------------------------------------------------------------------ setup
    def set_stage(self, stage: Stage) -> None:
        self.stage = stage
        self.arena.configure(stage)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.arena.randomise(rng)
        x, y, yaw = self.arena.spawn(rng)
        speed = rng.uniform(*self.stage.spawn_speed)
        self.state = self.model.spawn(x, y, yaw, speed)
        self.keeper.reset()
        self.step_count = 0
        self.stall_t = 0.0
        self.max_steps = int(self.stage.episode_seconds / self.car_cfg.dt)
        self._prev_gap = self._next_clip_gap()
        self.trail = [(x, y, yaw)]
        self.last_events = []
        return self._obs(), {}

    # -------------------------------------------------------------- observation
    def _next_clip_gap(self) -> float:
        """Distance to the nearest pillar that is not the one just clipped."""
        gaps = self.arena.pillar_gaps(self.state.x, self.state.y)
        if len(gaps) == 0:
            return 0.0
        last = self.keeper.combo.last_clip
        if 0 <= last < len(gaps) and len(gaps) > 1:
            gaps = np.delete(gaps, last)
        return float(np.min(gaps))

    def _obs(self) -> np.ndarray:
        s, cfg = self.state, self.car_cfg
        af, ar = self.model.slip_angles(s)
        kappa = self.model.slip_ratio(s)
        c = self.keeper.combo
        sc = self.score_cfg

        ego = [s.u / 20.0, s.v / 20.0, s.r / 2.5, math.sin(ar), math.cos(ar),
               math.sin(af), math.cos(af), np.clip(kappa / 2.0, -3, 3),
               s.steer / cfg.max_steer]

        combo = [1.0 if c.active else 0.0, c.multiplier / sc.mult_max,
                 c.time_mult / max(sc.mult_time_cap, 1e-6),
                 c.trick_mult / sc.mult_max, np.tanh(c.pending / 2000.0),
                 c.since_drift / sc.grace]

        beams = self.arena.beams(s.x, s.y, s.yaw) / self.arena.cfg.beam_range

        # Where the wall is, in body frame: radius fraction and outward normal.
        rad = math.hypot(s.x, s.y)
        out_ang = math.atan2(s.y, s.x) - s.yaw if rad > 1e-6 else 0.0
        wall = [rad / self.arena.cfg.radius, math.sin(out_ang), math.cos(out_ang),
                np.clip(self.arena.wall_clearance(s.x, s.y) / 20.0, 0.0, 3.0)]

        near = self.arena.nearest_pillars(s.x, s.y, s.yaw, N_PILLAR_SLOTS, c.last_clip)
        pil = np.column_stack([near[:, 0] / 40.0, near[:, 1] / 40.0,
                               near[:, 2] / 40.0, near[:, 3]]).ravel()

        return np.concatenate([ego, combo, beams, wall, pil]).astype(np.float32)

    # --------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steer, pedal, hand = float(a[0]), float(a[1]), float((a[2] + 1.0) * 0.5)

        self.state = self.model.step(self.state, steer, pedal, hand)
        s = self.state
        self.step_count += 1
        dt = self.car_cfg.dt
        rc = self.reward_cfg

        if not all(math.isfinite(q) for q in (s.x, s.y, s.u, s.v, s.r)):
            return self._obs(), -self.reward_cfg.crash_penalty, True, False, {"crashed": True,
                                                                              "events": []}

        crashed = self.arena.collided(s.x, s.y, s.yaw, self.car_cfg.body_len,
                                      self.car_cfg.body_wid)
        _af, ar = self.model.slip_angles(s)
        wall_dist = max(0.0, self.arena.wall_clearance(s.x, s.y))
        clip = self.arena.clip_index(s.x, s.y)

        earned, events = self.keeper.update(
            dt, slip=ar, speed=s.speed, yaw_rate=s.r, wall_dist=wall_dist,
            clip_hit=clip, x=s.x, y=s.y, crashed=crashed)
        self.last_events = events

        reward = rc.score_scale * earned
        reward += rc.speed_shaping * s.speed * dt
        reward += rc.slide_shaping * abs(ar) * s.speed * dt

        # Potential-based pull toward the next clip, so early policies find the
        # pillars at all. Zero-sum over a closed path, so it cannot be farmed.
        gap = self._next_clip_gap()
        if self.arena.n_pillars:
            reward += rc.pillar_shaping * (self._prev_gap - gap)
        self._prev_gap = gap

        terminated = False
        if crashed:
            reward -= rc.crash_penalty
            terminated = True

        if s.speed < self.score_cfg.spin_speed:
            self.stall_t += dt
            if self.stall_t > 3.0:
                reward -= rc.stall_penalty
                terminated = True
        else:
            self.stall_t = 0.0

        truncated = self.step_count >= self.max_steps
        if truncated or terminated:
            self.keeper.finish()

        self.trail.append((s.x, s.y, s.yaw))
        info = {"crashed": crashed, "events": events}
        if terminated or truncated:
            summary = self.keeper.summary()
            info.update(summary)
            # Normalised so the curriculum bar means the same thing at every
            # stage regardless of how long the episode is.
            info["score_rate"] = summary["score"] / self.stage.episode_seconds
            info["norm_score"] = float(np.tanh(info["score_rate"] / 400.0))
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def render(self):
        from drift.viewer import render_frame
        return render_frame(self)
