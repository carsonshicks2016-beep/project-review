"""ENDURANCE niche -- travel far on a finite energy reserve (ROADMAP 7.1).

The selective pressure is metabolic efficiency. Each control step burns energy
(a basal cost plus an activation cost that also produces heat); the reserve is
finite. While there is energy, forward progress is rewarded; once exhausted the
creature fatigues and its locomotion reward collapses to ~0. So a slow, cheap,
efficient gait that covers ground per joule beats a powerful but profligate one.
Escalation shrinks the effective reserve.

This task is stateful within an episode (`reset_episode` refills the reserve);
two runs with the same seed + actions deplete identically, so it stays
deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import NicheTask, register_task


@register_task
@dataclass
class EnduranceTask(NicheTask):
    name: str = "endurance"
    escalation_rate: float = 0.1
    max_steps: int = 1000
    forward_weight: float = 1.0
    alive_bonus: float = 0.05
    basal_cost: float = 0.5          # energy/s burned just being alive
    activation_cost: float = 0.05    # energy per unit sum(action^2) per second
    energy_budget: float = 60.0      # reserve (J-ish) at difficulty 0
    budget_per_difficulty: float = 0.5   # reserve shrinks by this fraction per unit

    def __post_init__(self):
        # per-episode reserve; a plain attribute so it stays out of the config dict
        self._energy = self._budget()

    def _budget(self) -> float:
        return self.energy_budget / (1.0 + self.budget_per_difficulty * self.difficulty)

    def reset_episode(self, rng):
        self._energy = self._budget()
        return self

    def reward(self, *, forward_vel, action, dt, **_):
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel())
        cost = (self.basal_cost + self.activation_cost * float(np.sum(a * a))) * dt
        self._energy = max(0.0, self._energy - cost)
        fatigue = 1.0 if self._energy > 0.0 else 0.0      # exhausted -> no progress reward
        fwd = self.forward_weight * float(np.nan_to_num(forward_vel)) * fatigue
        reward = fwd + self.alive_bonus * fatigue
        components = {"forward": fwd, "alive": self.alive_bonus * fatigue,
                      "energy_left": self._energy}
        return float(np.nan_to_num(reward)), components

    def is_fallen(self, *, root_z, stand_height, up_proj, **_):
        if not np.isfinite(root_z) or root_z < 0.5 * stand_height:
            return True, "low"
        if up_proj < 0.3:
            return True, "tipped"
        return False, ""
