"""A hand-written glideslope controller -- the benchmark the agent has to beat.

Phase 1 (acquire): fly to a hold point on the corridor axis, standing off the
keep-out sphere. Phase 2 (glideslope): walk down the axis with a commanded
closing rate that tapers with range. Both phases are velocity-tracking loops
with the Clohessy-Wiltshire drift cancelled out as feed-forward.
"""

from __future__ import annotations

import numpy as np


class GlideslopeController:
    def __init__(self, env, k_pos: float = 0.06, k_vel: float = 0.35, hold: float = 30.0):
        self.env = env
        self.k_pos = k_pos
        self.k_vel = k_vel
        self.hold = hold

    def __call__(self, obs=None) -> np.ndarray:
        env, t = self.env, self.env.task
        pos, vel = env.state[:3], env.state[3:]
        rng = float(np.linalg.norm(pos))
        axis = env.axis
        cos_a = float(np.dot(pos, axis) / rng) if rng > 1e-9 else 1.0

        if cos_a < env.cos_cone and rng > t.dock_radius:
            # Off-corridor: aim for the hold point rather than cutting the corner.
            target = axis * max(self.hold, min(rng, t.keepout_radius * 1.5))
            err = target - pos
            v_des = self.k_pos * err
            cap = t.v_cap_a + t.v_cap_b * float(np.linalg.norm(err))
        else:
            # On the corridor: glideslope straight in.
            v_des = -pos * self.k_pos
            cap = 0.85 * (t.v_cap_a + t.v_cap_b * rng)
            if rng < 3.0:
                cap = min(cap, 0.6 * t.dock_speed)

        speed = float(np.linalg.norm(v_des))
        if speed > cap:
            v_des *= cap / speed

        # Feed-forward the CW drift so the loop only has to fight the error.
        n = env.n
        drift = np.array([3 * n * n * pos[0] + 2 * n * vel[1], -2 * n * vel[0], -n * n * pos[2]])
        accel = self.k_vel * (v_des - vel) - drift
        return np.clip(accel / env.ship.max_accel, -1.0, 1.0)


def evaluate(env, policy, episodes: int = 200, seed: int = 0) -> dict:
    outcomes: dict[str, int] = {}
    dv, steps = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        info = {}
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        key = info.get("outcome") or "timeout"
        outcomes[key] = outcomes.get(key, 0) + 1
        if key == "docked":
            dv.append(info["dv_used"])
            steps.append(env.step_count)
    n = float(episodes)
    return {
        "success_rate": outcomes.get("docked", 0) / n,
        "outcomes": {k: v / n for k, v in sorted(outcomes.items())},
        "mean_dv": float(np.mean(dv)) if dv else float("nan"),
        "mean_time_s": float(np.mean(steps) * env.task.dt) if steps else float("nan"),
    }
