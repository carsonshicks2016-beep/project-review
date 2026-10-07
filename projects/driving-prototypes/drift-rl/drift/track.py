"""The arena: a walled circle with clip pillars, plus the sensing the car gets."""

from __future__ import annotations

import math

import numpy as np

from drift.config import ArenaConfig, Stage

# Pillar layouts per pillar count, in units of the arena radius. Rotation and a
# little jitter are randomised per episode so the policy learns the shape of the
# task rather than a fixed set of coordinates.
_LAYOUTS: dict[int, tuple[tuple[float, float], ...]] = {
    0: (),
    1: ((0.0, 0.0),),
    2: ((-0.26, 0.0), (0.26, 0.0)),
    3: ((0.0, 0.30), (-0.26, -0.15), (0.26, -0.15)),
    4: ((-0.28, -0.28), (0.28, -0.28), (0.28, 0.28), (-0.28, 0.28)),
    5: ((0.0, 0.0), (-0.30, -0.30), (0.30, -0.30), (0.30, 0.30), (-0.30, 0.30)),
}


class Arena:
    def __init__(self, cfg: ArenaConfig | None = None, stage: Stage | None = None) -> None:
        self.cfg = cfg or ArenaConfig()
        self.n_pillars = 0
        self.pillars = np.zeros((0, 2))
        self.beam_angles = np.linspace(-math.pi, math.pi, self.cfg.n_beams, endpoint=False)
        if stage is not None:
            self.configure(stage)

    def configure(self, stage: Stage) -> None:
        from dataclasses import replace
        self.cfg = replace(self.cfg, radius=stage.arena_radius)
        self.n_pillars = stage.n_pillars

    # -- layout ------------------------------------------------------------
    def randomise(self, rng: np.random.Generator) -> None:
        base = _LAYOUTS[self.n_pillars]
        if not base:
            self.pillars = np.zeros((0, 2))
            return
        theta = rng.uniform(0.0, 2 * math.pi)
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        pts = np.array(base) * self.cfg.radius
        pts = pts @ rot.T
        pts += rng.normal(0.0, 0.02 * self.cfg.radius, pts.shape)
        self.pillars = pts

    def spawn(self, rng: np.random.Generator) -> tuple[float, float, float]:
        """A start pose with room to work: away from the wall and the pillars."""
        for _ in range(64):
            rad = self.cfg.radius * math.sqrt(rng.uniform(0.0, 0.42))
            ang = rng.uniform(0.0, 2 * math.pi)
            x, y = rad * math.cos(ang), rad * math.sin(ang)
            if len(self.pillars) == 0 or np.min(np.hypot(*(self.pillars - (x, y)).T)) > 10.0:
                return x, y, rng.uniform(-math.pi, math.pi)
        return 0.0, 0.0, rng.uniform(-math.pi, math.pi)

    # -- queries -----------------------------------------------------------
    def wall_clearance(self, x: float, y: float) -> float:
        return self.cfg.radius - math.hypot(x, y)

    def pillar_gaps(self, x: float, y: float) -> np.ndarray:
        """Distance from the car centre to each pillar *surface*."""
        if len(self.pillars) == 0:
            return np.zeros(0)
        return np.hypot(*(self.pillars - (x, y)).T) - self.cfg.pillar_radius

    def clip_index(self, x: float, y: float) -> int:
        """Which pillar, if any, the car is close enough to be clipping."""
        gaps = self.pillar_gaps(x, y)
        if len(gaps) == 0:
            return -1
        i = int(np.argmin(gaps))
        return i if gaps[i] <= self.cfg.clip_band else -1

    def collided(self, x: float, y: float, yaw: float, body_len: float, body_wid: float) -> bool:
        """Rectangle-vs-wall on the body corners, circle-vs-circle on pillars."""
        hl, hw = body_len / 2, body_wid / 2
        c, s = math.cos(yaw), math.sin(yaw)
        for sx, sy in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
            cx, cy = x + sx * c - sy * s, y + sx * s + sy * c
            if math.hypot(cx, cy) >= self.cfg.radius:
                return True
        gaps = self.pillar_gaps(x, y)
        return bool(len(gaps)) and float(np.min(gaps)) <= hw

    def beams(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Rangefinder sweep, body-relative, clipped to ``beam_range``."""
        cfg = self.cfg
        out = np.empty(len(self.beam_angles))
        px, py = x, y
        pp = px * px + py * py
        for i, a in enumerate(self.beam_angles):
            th = yaw + a
            dx, dy = math.cos(th), math.sin(th)
            # Ray vs the arena wall from the inside: always exactly one hit.
            b = px * dx + py * dy
            t = -b + math.sqrt(max(0.0, b * b + cfg.radius * cfg.radius - pp))
            # Ray vs each pillar, keeping the nearest forward intersection.
            for cx, cy in self.pillars:
                mx, my = px - cx, py - cy
                bb = mx * dx + my * dy
                disc = bb * bb - (mx * mx + my * my - cfg.pillar_radius ** 2)
                if disc > 0.0:
                    tt = -bb - math.sqrt(disc)
                    if 0.0 < tt < t:
                        t = tt
            out[i] = min(t, cfg.beam_range)
        return out

    def nearest_pillars(self, x: float, y: float, yaw: float, k: int,
                        blocked: int = -1) -> np.ndarray:
        """The k nearest pillars in body frame: (dx, dy, dist, available)."""
        out = np.zeros((k, 4))
        if len(self.pillars) == 0:
            return out
        d = self.pillars - (x, y)
        dist = np.hypot(*d.T)
        order = np.argsort(dist)[:k]
        c, s = math.cos(-yaw), math.sin(-yaw)
        for j, i in enumerate(order):
            dx, dy = d[i]
            out[j] = (dx * c - dy * s, dx * s + dy * c, dist[i], 0.0 if i == blocked else 1.0)
        return out
