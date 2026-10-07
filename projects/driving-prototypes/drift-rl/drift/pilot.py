"""A scripted drift pilot: the baseline the learned policy has to beat.

It drives to a chosen equilibrium from ``drift.equilibria`` - feed-forward lock
and throttle straight off the manifold, plus feedback on slip angle, yaw rate
and wheel slip to stabilise a point that is open-loop unstable. It is a
competent drifter and a poor showman: it holds one angle and does not chain.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, replace

from drift.config import CarConfig
from drift.equilibria import Equilibrium, library
from drift.physics import CarModel, CarState

_LIB: dict[int, list[Equilibrium]] | None = None


def equilibrium_table() -> dict[int, list[Equilibrium]]:
    """Lazily built and cached; the solve costs a second or so."""
    global _LIB
    if _LIB is None:
        _LIB = library()
    return _LIB


def lookup(slip_deg: float, speed: float) -> Equilibrium:
    """Nearest slip-angle row, linearly interpolated along speed."""
    table = equilibrium_table()
    key = min(table, key=lambda a: abs(a - slip_deg))
    row = table[key]
    speeds = [e.speed for e in row]
    i = bisect.bisect_left(speeds, speed)
    if i <= 0:
        return row[0]
    if i >= len(row):
        return row[-1]
    lo, hi = row[i - 1], row[i]
    t = (speed - lo.speed) / (hi.speed - lo.speed)
    blend = lambda p: (1 - t) * getattr(lo, p) + t * getattr(hi, p)
    return replace(lo, speed=speed, beta=blend("beta"), yaw_rate=blend("yaw_rate"),
                   steer=blend("steer"), throttle=blend("throttle"),
                   kappa=blend("kappa"), radius=blend("radius"))


@dataclass
class PilotGains:
    k_slip: float = 1.3      # rad of extra lock per rad of slip error
    k_yaw: float = 0.15      # rad of lock per rad/s of yaw-rate error
    k_kappa: float = 1.2     # throttle per unit slip-ratio error
    k_speed: float = 0.15   # slip-ratio bias per m/s of speed error


class DriftPilot:
    """Entry -> hold. Optionally flips direction on a timer to chain switchbacks."""

    def __init__(self, cfg: CarConfig | None = None, slip_deg: float = 35.0,
                 speed: float = 16.0, direction: int = 1, flip_every: float | None = None,
                 gains: PilotGains | None = None, arena=None, wall_margin: float = 24.0) -> None:
        self.cfg = cfg or CarConfig()
        self.model = CarModel(self.cfg)
        self.slip_deg = slip_deg
        self.speed_ref = speed
        self.d = direction                  # +1 left-hand drift, -1 right
        self.flip_every = flip_every
        self.gains = gains or PilotGains()
        self.arena = arena              # optional: lets it switchback off the wall
        self.wall_margin = wall_margin
        self.phase = "launch"
        self.t = 0.0
        self.phase_t = 0.0

    def reset(self) -> None:
        self.phase, self.t, self.phase_t = "launch", 0.0, 0.0

    def _set_phase(self, name: str) -> None:
        if name != self.phase:
            self.phase, self.phase_t = name, 0.0

    def _aim_inward(self, s: CarState) -> float:
        """Steer toward the middle of the arena; used while building speed."""
        if self.arena is None:
            return 0.0
        err = math.atan2(-s.y, -s.x) - s.yaw
        err = (err + math.pi) % (2 * math.pi) - math.pi
        return max(-1.0, min(1.0, 1.8 * err))

    def act(self, s: CarState) -> tuple[float, float, float]:
        c = self.cfg
        dt = c.dt
        self.t += dt
        self.phase_t += dt
        _af, ar = self.model.slip_angles(s)
        kappa = self.model.slip_ratio(s)
        drifting = abs(ar) > math.radians(14.0) and s.speed > 6.0

        # Wall guard: pick the slide direction that curves the velocity back
        # toward the middle. A fixed-radius drift otherwise ends in the fence.
        if self.arena is not None and self.phase == "hold" and self.phase_t > 0.6:
            clear = self.arena.wall_clearance(s.x, s.y)
            if clear < self.wall_margin:
                course = s.yaw + s.beta
                vx, vy = math.cos(course), math.sin(course)
                n = math.hypot(s.x, s.y) or 1.0
                tx, ty = -s.x / n, -s.y / n          # unit vector to the middle
                want = 1 if (vx * ty - vy * tx) > 0 else -1
                if want != self.d:
                    self.d = want
                    self.phase_t = 0.0

        if self.flip_every and self.phase == "hold" and self.phase_t > self.flip_every:
            # A real switchback keeps the rear loose through the transition:
            # mirror the equilibrium target and let the feedback swing it over.
            # No handbrake, no lift - either would drop the combo.
            self.d = -self.d
            self.phase_t = 0.0

        if s.speed < 0.55 * self.speed_ref and self.phase != "launch":
            self._set_phase("launch")      # bogged down; go get some speed back

        if self.phase == "launch":
            if s.speed >= self.speed_ref * 0.92:
                self._set_phase("flick")
            return self._aim_inward(s), 1.0, 0.0

        if self.phase == "flick":
            # Weight transfer the wrong way, then snap in and lock the rear.
            if self.phase_t < 0.16:
                return -0.35 * self.d, 0.0, 0.0
            hand = 1.0 if self.phase_t < 0.42 else 0.0
            if drifting and math.copysign(1, ar) == -self.d and self.phase_t > 0.3:
                self._set_phase("hold")
            elif self.phase_t > 1.0:
                self._set_phase("hold")
            return 0.75 * self.d, 0.15, hand

        # hold: ride the equilibrium
        eq = lookup(self.slip_deg, max(s.speed, 8.0))
        ar_ref = -self.d * abs(eq.slip_rear)
        r_ref = self.d * abs(eq.yaw_rate) * (s.speed / max(eq.speed, 1.0))
        steer_ff = -self.d * abs(eq.steer)

        g = self.gains
        delta = steer_ff + g.k_slip * (ar - ar_ref) - g.k_yaw * (s.r - r_ref)
        steer = max(-1.0, min(1.0, delta / c.max_steer))

        kap_ref = abs(eq.kappa) + g.k_speed * (self.speed_ref - s.speed)
        kap_ref = max(0.05, min(2.0, kap_ref))
        thr = eq.throttle + g.k_kappa * (kap_ref - kappa)
        thr = max(-0.4, min(1.0, thr))

        if not drifting and self.phase_t > 1.0:
            self._set_phase("flick")
        return steer, thr, 0.0
