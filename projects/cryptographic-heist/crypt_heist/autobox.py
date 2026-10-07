"""Automatic gearbox helper adapted from the Supra driving app."""

from __future__ import annotations

from .config import CarSpec
from .physics import Vehicle


class AutoBox:
    """Small automatic transmission controller for scripted and AI drivers."""

    def __init__(self, spec: CarSpec):
        self.spec = spec
        self.shift_cooldown = 0.0
        self.clutch = 0.0

    def update(self, veh: Vehicle, throttle: float, dt: float):
        """Return ``(clutch, shift_up, shift_down)`` for the next physics step."""
        self.shift_cooldown = max(0.0, self.shift_cooldown - dt)
        up = down = False
        rpm = veh.rpm
        if self.shift_cooldown == 0.0:
            if rpm > self.spec.redline_rpm * 0.95 and veh.gear < len(self.spec.gear_ratios):
                up = True
                self.shift_cooldown = 0.34
            elif rpm < 1450 and veh.gear > 1 and veh.speed > 1.0:
                down = True
                self.shift_cooldown = 0.34

        idle_thresh = self.spec.idle_rpm * 1.25
        if up or down:
            target = 0.22
        elif throttle < 0.05 and (veh.speed < 1.5 or rpm < idle_thresh):
            target = 0.0
        else:
            target = 1.0
        self.clutch += (target - self.clutch) * min(1.0, dt * 12.0)
        return self.clutch, up, down
