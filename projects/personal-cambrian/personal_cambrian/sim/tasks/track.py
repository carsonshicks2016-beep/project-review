"""TRACK (sprint) niche -- raw forward speed against a headwind (ROADMAP 7.1).

A locomotion variant tuned for sprinting: forward speed is weighted heavily and
the alive bonus is small, so the selective pressure is "go fast", not "don't
fall". Escalation ramps a headwind (a constant external force opposing forward
motion); a faster, more powerful runner overcomes it, a plodder stalls.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import register_task
from .locomotion import LocomotionTask

# body index 1 is the creature root (body 0 is the world); xfrc_applied is global.
_ROOT_BODY = 1


@register_task
@dataclass
class TrackTask(LocomotionTask):
    name: str = "track"
    escalation_rate: float = 0.1
    forward_weight: float = 2.5      # sprint: speed dominates
    alive_bonus: float = 0.05        # barely reward survival
    ctrl_cost: float = 5e-4
    headwind_per_difficulty: float = 8.0   # N of opposing force per difficulty unit

    def perturb(self, model, data, rng, elapsed: int):
        wind = self.headwind_per_difficulty * self.difficulty
        if wind != 0.0:
            data.xfrc_applied[_ROOT_BODY] = 0.0
            data.xfrc_applied[_ROOT_BODY, self.forward_axis] = -wind
