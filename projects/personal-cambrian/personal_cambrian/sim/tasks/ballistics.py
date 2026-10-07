"""BALLISTICS niche -- launch the body, maximize takeoff (ROADMAP 7.1).

The selective pressure is explosive power: reward the upward velocity of the
centre of mass and any height gained above the standing pose. A creature that
builds a coiled, spring-loaded plan and fires it skyward wins; one that just
shuffles scores ~0. Episodes are short (a jump is brief) and the body is NOT
penalised for leaving the ground. Escalation raises effective gravity, so a
stronger launch is needed to clear the same bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import NicheTask, register_task

_BASE_G = 9.81


@register_task
@dataclass
class BallisticsTask(NicheTask):
    name: str = "ballistics"
    escalation_rate: float = 0.1
    max_steps: int = 150
    launch_weight: float = 1.0       # reward per m/s of upward COM velocity
    height_weight: float = 4.0       # reward per m of COM height above standing
    ctrl_cost: float = 1e-4
    gravity_per_difficulty: float = 0.5   # +50% g per difficulty unit

    def configure_model(self, model):
        g = _BASE_G * (1.0 + self.gravity_per_difficulty * self.difficulty)
        model.opt.gravity[2] = -g
        return model

    def reward(self, *, com_vz, com_z, stand_height, action, dt, **_):
        vz = float(np.nan_to_num(com_vz))
        gain = max(0.0, float(np.nan_to_num(com_z)) - float(stand_height))
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel())
        launch = self.launch_weight * max(0.0, vz)
        height = self.height_weight * gain
        energy = self.ctrl_cost * float(np.sum(a * a)) * dt
        reward = launch + height - energy
        components = {"launch": launch, "height": height, "energy": -energy}
        return float(np.nan_to_num(reward)), components

    def is_fallen(self, *, root_z, stand_height, up_proj, **_):
        # a jump leaves the ground on purpose -> never terminate on "airborne".
        return False, ""
