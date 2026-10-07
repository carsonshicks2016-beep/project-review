"""CHAOS GRID niche -- stay upright under random shoves (ROADMAP 7.1).

The selective pressure is balance and robustness. At intervals a random
horizontal impulse is applied to the body; the creature is rewarded for staying
upright and near its starting spot, and a fall ends the episode. Low, wide,
well-damped body plans persist; tall narrow ones topple. Escalation increases the
push magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import NicheTask, register_task

_ROOT_BODY = 1


@register_task
@dataclass
class ChaosGridTask(NicheTask):
    name: str = "chaos_grid"
    escalation_rate: float = 0.1
    max_steps: int = 500
    alive_bonus: float = 1.0
    drift_cost: float = 0.5          # penalty per metre shoved from the start spot
    tip_cost: float = 1.0            # penalty for leaning (1 - up_proj)
    ctrl_cost: float = 1e-4
    fall_fraction: float = 0.5
    upright_min: float = 0.3
    push_period: int = 25            # apply a shove every N control steps
    push_base: float = 30.0          # N of impulse at difficulty 0
    push_per_difficulty: float = 40.0

    def perturb(self, model, data, rng, elapsed: int):
        data.xfrc_applied[_ROOT_BODY] = 0.0
        if self.push_period > 0 and elapsed % self.push_period == 0:
            mag = self.push_base + self.push_per_difficulty * self.difficulty
            direction = rng.uniform(-1.0, 1.0, size=2)
            n = np.linalg.norm(direction)
            if n > 0:
                data.xfrc_applied[_ROOT_BODY, :2] = mag * direction / n

    def reward(self, *, displacement, up_proj, action, dt, **_):
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel())
        drift = self.drift_cost * float(np.nan_to_num(displacement))
        tip = self.tip_cost * (1.0 - float(np.nan_to_num(up_proj)))
        energy = self.ctrl_cost * float(np.sum(a * a)) * dt
        reward = self.alive_bonus - drift - tip - energy
        components = {"alive": self.alive_bonus, "drift": -drift,
                      "tip": -tip, "energy": -energy}
        return float(np.nan_to_num(reward)), components

    def is_fallen(self, *, root_z, stand_height, up_proj, **_):
        if not np.isfinite(root_z) or root_z < self.fall_fraction * stand_height:
            return True, "low"
        if up_proj < self.upright_min:
            return True, "tipped"
        return False, ""
