"""Locomotion task: travel along an axis without falling (ROADMAP Stage 2.4 / 2.5).

The reward is the selective pressure that Stage-3 RL and all of evolution optimize
against, so it is kept small, signed, and inspectable:

    reward =  forward_weight * forward_velocity        (progress)
            - ctrl_cost      * sum(action^2) * dt        (energy; ties to the
                                                          metabolic budget, Stage 6)
            - smooth_cost    * sum(d_action^2)           (penalize jerky control)
            + alive_bonus                                (encourage not falling)

Termination is a fall: the root drops below a fraction of its standing height, or
the torso tips past an uprightness threshold (so a creature can't "win" by diving
forward onto its face). The reward and fall test are pure (no MuJoCo handle), so
they are unit-testable in isolation. The task is a declarative `NicheTask` config
(serializable via to_dict/from_dict; escalation hook for the Stage-8 curriculum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import NicheTask, register_task


@register_task
@dataclass
class LocomotionTask(NicheTask):
    name: str = "locomotion"
    forward_weight: float = 1.0
    ctrl_cost: float = 1e-3        # energy penalty on actuation
    smooth_cost: float = 1e-3      # penalty on change in action
    alive_bonus: float = 0.1
    fall_fraction: float = 0.5     # fall if root height < fraction * standing height
    upright_min: float = 0.3       # fall if torso up-axis . world-up < this (~72 deg)

    def reward(self, *, forward_vel: float, action, prev_action, dt: float, **_):
        """Return (reward, components). Always finite (NaN inputs -> 0)."""
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel(), nan=0.0)
        pa = np.nan_to_num(np.asarray(prev_action, dtype=np.float64).ravel(), nan=0.0)
        fwd = self.forward_weight * float(np.nan_to_num(forward_vel))
        energy = self.ctrl_cost * float(np.sum(a * a)) * dt
        smooth = self.smooth_cost * float(np.sum((a - pa) ** 2))
        reward = fwd - energy - smooth + self.alive_bonus
        components = {"forward": fwd, "energy": -energy,
                      "smooth": -smooth, "alive": self.alive_bonus}
        return float(np.nan_to_num(reward)), components

    def is_fallen(self, *, root_z: float, stand_height: float, up_proj: float, **_):
        """Return (fallen, reason). `up_proj` is the torso up-axis projected onto
        world up (1 = upright, 0 = on its side, -1 = inverted)."""
        if not np.isfinite(root_z) or root_z < self.fall_fraction * stand_height:
            return True, "low"
        if up_proj < self.upright_min:
            return True, "tipped"
        return False, ""
