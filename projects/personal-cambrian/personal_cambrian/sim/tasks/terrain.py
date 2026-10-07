"""TERRAIN niche -- locomote up a slope (ROADMAP 7.1).

The selective pressure is climbing on an incline. Rather than rebuild the floor
as a heightfield, the slope is applied as *equivalent gravity*: on an incline of
angle theta, a body on flat ground with gravity tilted by theta feels exactly the
same forces as on the real slope. Gravity gains a horizontal component pulling
"downhill" (toward -forward), so forward progress is uphill work. Escalation
steepens the slope.

[APPROX] This models a uniform slope, not bumps/obstacles; heightfield terrain is
a later refinement. Reward is forward (uphill) locomotion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .base import register_task
from .locomotion import LocomotionTask

_BASE_G = 9.81


@register_task
@dataclass
class TerrainTask(LocomotionTask):
    name: str = "terrain"
    escalation_rate: float = 0.1
    terrain: str = "slope"
    forward_weight: float = 1.5      # reward climbing
    max_slope_rad: float = 0.7       # ~40 deg at difficulty 1.0
    slope_per_difficulty: float = 0.35

    def _slope(self) -> float:
        return min(self.max_slope_rad, self.slope_per_difficulty * self.difficulty)

    def configure_model(self, model):
        theta = self._slope()
        # forward (+axis) is uphill: gravity pulls back toward -axis, down on z
        model.opt.gravity[self.forward_axis] = -_BASE_G * math.sin(theta)
        model.opt.gravity[2] = -_BASE_G * math.cos(theta)
        return model
