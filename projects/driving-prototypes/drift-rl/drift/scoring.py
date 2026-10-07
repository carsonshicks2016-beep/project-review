"""Arcade trick scoring: angle x speed x time, multiplied by whatever you chain.

The rule the whole project turns on: points accrue while sliding, but they sit
in an *unbanked* pool that keeps growing a multiplier as long as the combo
stays alive. Straighten up cleanly and the pool banks. Spin, bog down or hit
something and you lose most of it. That is what makes "chain tricks together"
a different problem from "hold one big drift".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from drift.config import DEG, ScoreConfig

TRICKS = ("SWITCHBACK", "DONUT", "CLIP", "MANJI", "WALL KISS", "BIG ANGLE")


@dataclass
class Event:
    name: str
    points: float          # already multiplied
    multiplier: float
    x: float = 0.0
    y: float = 0.0


@dataclass
class ComboState:
    active: bool = False
    duration: float = 0.0        # s of live combo
    pending: float = 0.0         # points not yet banked
    time_mult: float = 0.0       # multiplier earned by staying sideways (capped)
    trick_mult: float = 0.0      # multiplier earned by linking tricks
    mult_cap: float = 12.0
    side_time: float = 0.0       # s on the current side of the slide
    side_peak: float = 0.0       # biggest slip angle reached on this side
    direction: int = 0           # sign of the current slide, 0 if none
    since_drift: float = 0.0     # s spent below the drift threshold
    yaw_accum: float = 0.0       # rad of rotation this slide, for donuts
    switchbacks: int = 0
    switchback_times: list[float] = field(default_factory=list)
    big_angle_t: float = 0.0
    last_wall_kiss: float = -99.0
    last_clip: int = -1          # pillar index, so you must alternate

    @property
    def multiplier(self) -> float:
        return min(self.mult_cap, 1.0 + self.time_mult + self.trick_mult)


class ScoreKeeper:
    """Consumes vehicle state each step and emits points plus trick events."""

    def __init__(self, cfg: ScoreConfig | None = None) -> None:
        self.cfg = cfg or ScoreConfig()
        self.reset()

    def reset(self) -> None:
        self.combo = ComboState(mult_cap=self.cfg.mult_max)
        self._earned = 0.0
        self.banked = 0.0
        self.t = 0.0
        self.best_combo = 0.0
        self.best_multiplier = 1.0
        self.trick_counts = {name: 0 for name in TRICKS}
        self.events: list[Event] = []
        self.wipeouts = 0

    # -- helpers -----------------------------------------------------------
    @property
    def total(self) -> float:
        """Banked plus what is still riding on the live combo."""
        return self.banked + self.combo.pending

    def _award(self, name: str, base: float, mult_gain: float, x: float, y: float) -> float:
        """Fire a trick bonus. The points are returned so they reach the reward:
        a bonus that only shows up in the score is invisible to the agent."""
        c = self.combo
        pts = base * c.multiplier
        c.pending += pts
        self._earned += pts
        c.trick_mult = min(self.cfg.mult_max - 1.0, c.trick_mult + mult_gain)
        self.trick_counts[name] += 1
        self.events.append(Event(name, pts, c.multiplier, x, y))
        self.best_multiplier = max(self.best_multiplier, c.multiplier)
        return pts

    def _bank(self) -> float:
        c = self.combo
        got = c.pending
        self.banked += got
        self.best_combo = max(self.best_combo, got)
        self.combo = ComboState(mult_cap=self.cfg.mult_max)
        return got

    def _wipe(self) -> float:
        """Lose the un-banked pool. Returns the points forfeited."""
        c = self.combo
        lost = c.pending * self.cfg.spin_pending_frac
        self.banked += c.pending - lost
        self.best_combo = max(self.best_combo, c.pending)
        self.combo = ComboState(mult_cap=self.cfg.mult_max)
        self.wipeouts += 1
        return lost

    # -- main update -------------------------------------------------------
    def update(self, dt: float, *, slip: float, speed: float, yaw_rate: float,
               wall_dist: float, clip_hit: int = -1, x: float = 0.0, y: float = 0.0,
               crashed: bool = False) -> tuple[float, list[Event]]:
        """Advance one step.

        Returns (points earned this step, events fired this step). ``clip_hit``
        is the index of a pillar being clipped, or -1.
        """
        cfg = self.cfg
        c = self.combo
        self.t += dt
        self.events = []
        self._earned = 0.0

        if crashed:
            return -self._wipe(), self.events

        drifting = abs(slip) >= cfg.angle_min and speed >= cfg.speed_min
        spun = abs(slip) >= cfg.spin_angle or (c.active and speed < cfg.spin_speed)

        if spun and c.active:
            return -self._wipe(), self.events

        if drifting:
            d = 1 if slip > 0 else -1
            if not c.active:
                c.active, c.direction = True, d
            elif d != c.direction:
                # Rear swung through zero and back out the other side. Only a
                # slide that was properly established counts as a switchback.
                real = (c.side_peak >= cfg.switchback_min_angle
                        and c.side_time >= cfg.switchback_min_time)
                c.direction = d
                c.yaw_accum = 0.0
                c.side_time = 0.0
                c.side_peak = 0.0
                if real:
                    c.switchbacks += 1
                    c.switchback_times.append(self.t)
                    self._award("SWITCHBACK", cfg.switchback_bonus, cfg.switchback_mult, x, y)
                    recent = [t for t in c.switchback_times if self.t - t <= cfg.manji_window]
                    c.switchback_times = recent
                    if len(recent) >= 3:
                        c.switchback_times = []
                        self._award("MANJI", cfg.manji_bonus, cfg.switchback_mult, x, y)

            c.since_drift = 0.0
            c.duration += dt
            c.side_time += dt
            c.side_peak = max(c.side_peak, abs(slip))

            # Base flow: degrees of slip x m/s x seconds, then multiplied.
            rate = cfg.base_rate * math.degrees(abs(slip)) * speed
            step_pts = rate * dt * c.multiplier
            c.pending += step_pts
            self._earned += step_pts
            c.time_mult = min(cfg.mult_time_cap, c.time_mult + cfg.mult_per_second * dt)
            self.best_multiplier = max(self.best_multiplier, c.multiplier)

            # Donut: a full turn without swapping direction.
            c.yaw_accum += abs(yaw_rate) * dt
            if c.yaw_accum >= 2 * math.pi:
                c.yaw_accum -= 2 * math.pi
                self._award("DONUT", cfg.donut_bonus, cfg.donut_mult, x, y)

            # Clip: you must alternate pillars, so a figure-8 scores and a
            # single endless donut around one cone does not.
            if clip_hit >= 0 and clip_hit != c.last_clip:
                c.last_clip = clip_hit
                self._award("CLIP", cfg.clip_bonus, cfg.clip_mult, x, y)

            if wall_dist <= cfg.wall_kiss_dist and self.t - c.last_wall_kiss > cfg.wall_kiss_cooldown:
                c.last_wall_kiss = self.t
                self._award("WALL KISS", cfg.wall_kiss_bonus, 0.25, x, y)

            if abs(slip) >= cfg.big_angle:
                c.big_angle_t += dt
                if c.big_angle_t >= cfg.big_angle_hold:
                    c.big_angle_t = 0.0
                    self._award("BIG ANGLE", cfg.big_angle_bonus, 0.3, x, y)
            else:
                c.big_angle_t = 0.0

        elif c.active:
            # Grace window: a quick straighten between links does not break it.
            c.since_drift += dt
            c.duration += dt
            if c.since_drift > cfg.grace:
                self._bank()

        return self._earned, self.events

    def finish(self) -> None:
        """Episode over: a live combo banks (you kept it clean to the flag)."""
        if self.combo.active:
            self._bank()

    def summary(self) -> dict:
        return {
            "score": self.banked,
            "best_combo": self.best_combo,
            "best_multiplier": self.best_multiplier,
            "wipeouts": self.wipeouts,
            **{f"n_{k.lower().replace(' ', '_')}": v for k, v in self.trick_counts.items()},
        }
