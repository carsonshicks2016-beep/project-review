"""Gymnasium environment: rendezvous and dock with a target in circular LEO."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import CURRICULUM, ShipConfig, Stage, TaskConfig, stage_task
from .dynamics import mean_motion, rk4_step


class DockingEnv(gym.Env):
    """Continuous-thrust proximity operations in the target's Hill frame.

    The chaser starts somewhere on a shell around the target and must arrive at
    the docking port at the origin: inside the approach cone, inside the capture
    radius, and slow. The observation is what a real rangefinder / docking
    camera pair would give you -- relative state plus a few derived cues -- and
    the action is a throttle command per body axis.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: TaskConfig | None = None,
        ship: ShipConfig | None = None,
        stage: Stage | None = None,
    ):
        super().__init__()
        self.stage = stage if stage is not None else CURRICULUM[-1]
        base = task if task is not None else TaskConfig()
        self.task = stage_task(base, self.stage)
        self.ship = ship if ship is not None else ShipConfig()

        self.n = mean_motion(self.task.altitude)
        self.axis = np.array(self.task.port_axis, dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)
        self.cos_cone = np.cos(np.radians(self.task.cone_half_angle_deg))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

        self.state = np.zeros(6)
        self.dv_used = 0.0
        self.step_count = 0
        self._trajectory: list[np.ndarray] = []

    # ------------------------------------------------------------------ setup

    def set_stage(self, stage: Stage, base: TaskConfig | None = None) -> None:
        self.stage = stage
        self.task = stage_task(base if base is not None else self.task, stage)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        st = self.stage

        # Spawn on a shell, biased towards the port side by the stage's cone.
        r = rng.uniform(st.range_min, st.range_max)
        direction = self._sample_direction(rng, np.radians(st.spawn_cone_deg))
        pos = r * direction
        vel = rng.normal(0.0, st.vel_sigma, size=3)

        self.state = np.concatenate([pos, vel])
        self.dv_used = 0.0
        self.step_count = 0
        self._trajectory = [self.state.copy()]
        return self._observe(), self._info(0.0)

    def _sample_direction(self, rng, half_angle: float) -> np.ndarray:
        """Unit vector within `half_angle` of the port axis (uniform on the cap)."""
        cos_min = np.cos(min(half_angle, np.pi))
        c = rng.uniform(cos_min, 1.0)
        s = np.sqrt(max(0.0, 1.0 - c * c))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        # Build an orthonormal basis around the port axis.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, self.axis)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        u = np.cross(self.axis, tmp)
        u /= np.linalg.norm(u)
        w = np.cross(self.axis, u)
        return c * self.axis + s * (np.cos(phi) * u + np.sin(phi) * w)

    # ------------------------------------------------------------------- step

    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        act = np.where(np.abs(act) < self.ship.min_throttle, 0.0, act)

        dt = self.task.dt
        accel = act * self.ship.max_accel

        # Honour the remaining delta-v budget rather than letting it go negative.
        dv_cmd = np.linalg.norm(accel) * dt
        remaining = self.ship.dv_budget - self.dv_used
        if dv_cmd > remaining > 0.0:
            accel *= remaining / dv_cmd
            dv_cmd = remaining
        elif remaining <= 0.0:
            accel = np.zeros(3)
            dv_cmd = 0.0

        prev = self.state.copy()
        self.state = rk4_step(self.state, accel, self.n, dt)
        self.dv_used += dv_cmd
        self.step_count += 1
        self._trajectory.append(self.state.copy())

        reward, terminated, truncated, outcome = self._reward(prev, self.state, dv_cmd)
        return self._observe(), float(reward), terminated, truncated, self._info(reward, outcome)

    # ----------------------------------------------------------------- reward

    def _potential(self, state: np.ndarray) -> float:
        """Shaping potential; differences of this are what the agent is paid."""
        t = self.task
        rng_ = np.linalg.norm(state[:3])
        spd = np.linalg.norm(state[3:])
        # log range keeps the gradient meaningful across three orders of magnitude
        return -(t.w_range * np.log1p(rng_) + t.w_speed * spd / (1.0 + rng_))

    def _reward(self, prev, cur, dv_cmd):
        t = self.task
        pos, vel = cur[:3], cur[3:]
        rng_ = float(np.linalg.norm(pos))
        spd = float(np.linalg.norm(vel))

        reward = self._potential(cur) - self._potential(prev)
        reward -= t.w_fuel * dv_cmd

        # Safety envelope: don't barrel in fast.
        v_cap = t.v_cap_a + t.v_cap_b * rng_
        if spd > v_cap:
            reward -= t.w_vcap * (spd - v_cap)

        cos_a = float(np.dot(pos, self.axis) / rng_) if rng_ > 1e-9 else 1.0
        in_cone = cos_a >= self.cos_cone

        # Inside the keep-out sphere, being off the corridor axis is penalised
        # continuously and, once close, is treated as a structural hit.
        if rng_ < t.keepout_radius and not in_cone:
            reward -= t.w_cone * (self.cos_cone - cos_a) * (t.keepout_radius - rng_)

        outcome = None
        terminated = truncated = False

        lateral = vel - np.dot(vel, self.axis) * self.axis
        if rng_ <= t.dock_radius:
            terminated = True
            captured = (
                in_cone
                and spd <= t.dock_speed
                and float(np.linalg.norm(lateral)) <= t.dock_lateral_speed
            )
            if captured:
                thrift = 1.0 - self.dv_used / self.ship.dv_budget
                reward += t.r_dock * (1.0 + 0.5 * thrift)
                outcome = "docked"
            else:
                # Graded, not binary: arriving slightly hot must still beat never
                # arriving, or the agent learns to loiter outside the corridor.
                excess = max(
                    spd / t.dock_speed,
                    float(np.linalg.norm(lateral)) / t.dock_lateral_speed,
                    1.0,
                )
                severity = min(1.0, (excess - 1.0) / 5.0)
                if not in_cone:
                    severity = max(severity, 0.5 + 0.5 * min(1.0, self.cos_cone - cos_a))
                reward += t.r_crash * severity
                outcome = "hard_contact"
        elif rng_ < 0.35 * t.keepout_radius and not in_cone:
            reward += t.r_crash
            outcome = "keepout_violation"
            terminated = True
        elif rng_ > t.bounds:
            reward += t.r_abort
            outcome = "out_of_bounds"
            terminated = True
        elif self.dv_used >= self.ship.dv_budget and rng_ > t.keepout_radius:
            reward += t.r_abort
            outcome = "out_of_fuel"
            terminated = True
        elif self.step_count >= t.max_steps:
            truncated = True
            outcome = "timeout"

        return reward, terminated, truncated, outcome

    # ------------------------------------------------------------ observation

    def _observe(self) -> np.ndarray:
        t = self.task
        pos, vel = self.state[:3], self.state[3:]
        rng_ = float(np.linalg.norm(pos))
        unit = pos / rng_ if rng_ > 1e-9 else np.zeros(3)
        closing = float(np.dot(vel, unit))  # >0 means opening
        spd = float(np.linalg.norm(vel))
        v_cap = t.v_cap_a + t.v_cap_b * rng_
        obs = np.concatenate(
            [
                pos / t.len_scale,
                vel / t.vel_scale,
                unit,
                [
                    np.log1p(rng_) / 7.0,
                    closing / t.vel_scale,
                    float(np.dot(unit, self.axis)),          # corridor alignment
                    (spd - v_cap) / t.vel_scale,             # envelope margin
                    1.0 - self.dv_used / self.ship.dv_budget,
                    1.0 - self.step_count / t.max_steps,
                ],
            ]
        )
        return obs.astype(np.float32)

    def _info(self, reward: float, outcome: str | None = None) -> dict:
        pos = self.state[:3]
        return {
            "range": float(np.linalg.norm(pos)),
            "speed": float(np.linalg.norm(self.state[3:])),
            "dv_used": float(self.dv_used),
            "outcome": outcome,
            "is_success": outcome == "docked",
        }

    @property
    def trajectory(self) -> np.ndarray:
        return np.array(self._trajectory)
