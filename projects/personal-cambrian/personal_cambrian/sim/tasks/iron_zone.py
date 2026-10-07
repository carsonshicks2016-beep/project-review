"""IRON ZONE niche -- brace and hold against a load, maximize force (ROADMAP 7.1).

The selective pressure is static strength: the creature must generate large
actuator forces while *staying put* against an external load that tries to drag
it away. Reward rises with force output and falls with how far it is dragged, so
a strong, well-anchored body plan (big muscles, low stance, wide base) wins and a
weak one is pushed out of the zone. Escalation increases the load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import NicheTask, register_task

_ROOT_BODY = 1


@register_task
@dataclass
class IronZoneTask(NicheTask):
    name: str = "iron_zone"
    escalation_rate: float = 0.1
    max_steps: int = 400
    strength_weight: float = 1.0
    force_scale: float = 200.0       # N that maps to ~1.0 of reward
    drift_cost: float = 2.0          # penalty per metre dragged from the zone
    ctrl_cost: float = 1e-4
    alive_bonus: float = 0.1
    fall_fraction: float = 0.5
    upright_min: float = 0.3
    load_base: float = 40.0          # N of pull even at difficulty 0
    load_per_difficulty: float = 60.0

    def perturb(self, model, data, rng, elapsed: int):
        load = self.load_base + self.load_per_difficulty * self.difficulty
        data.xfrc_applied[_ROOT_BODY] = 0.0
        data.xfrc_applied[_ROOT_BODY, self.forward_axis] = load   # drag along +axis

    def reward(self, *, actuator_force, displacement, action, dt, **_):
        f = np.nan_to_num(np.asarray(actuator_force, dtype=np.float64).ravel())
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel())
        effort = self.strength_weight * float(np.mean(np.abs(f))) / self.force_scale
        drift = self.drift_cost * float(np.nan_to_num(displacement))
        energy = self.ctrl_cost * float(np.sum(a * a)) * dt
        reward = effort - drift - energy + self.alive_bonus
        components = {"effort": effort, "drift": -drift,
                      "energy": -energy, "alive": self.alive_bonus}
        return float(np.nan_to_num(reward)), components

    def is_fallen(self, *, root_z, stand_height, up_proj, **_):
        if not np.isfinite(root_z) or root_z < self.fall_fraction * stand_height:
            return True, "low"
        if up_proj < self.upright_min:
            return True, "tipped"
        return False, ""
