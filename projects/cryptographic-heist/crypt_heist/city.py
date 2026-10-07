"""Procedural urban grid and simple building collision response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import Vehicle


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def expanded(self, pad: float) -> "Rect":
        return Rect(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class UrbanGrid:
    """Block-and-road city used by both rendering and the MARL environment."""

    def __init__(
        self,
        cols: int = 8,
        rows: int = 6,
        block: float = 78.0,
        road: float = 26.0,
        seed: int | None = 7,
    ):
        self.cols = cols
        self.rows = rows
        self.block = float(block)
        self.road = float(road)
        self.pitch = self.block + self.road
        self.width = cols * self.block + (cols + 1) * self.road
        self.height = rows * self.block + (rows + 1) * self.road
        self.left = -self.width / 2.0
        self.top = -self.height / 2.0
        self.right = self.width / 2.0
        self.bottom = self.height / 2.0
        self.rng = np.random.default_rng(seed)
        self.buildings = self._make_buildings()
        self.vertical_roads = [
            self.left + self.road / 2.0 + i * self.pitch for i in range(cols + 1)
        ]
        self.horizontal_roads = [
            self.top + self.road / 2.0 + i * self.pitch for i in range(rows + 1)
        ]

    def _make_buildings(self) -> list[Rect]:
        buildings = []
        for r in range(self.rows):
            y0 = self.top + self.road + r * self.pitch
            for c in range(self.cols):
                x0 = self.left + self.road + c * self.pitch
                buildings.append(Rect(x0, y0, x0 + self.block, y0 + self.block))
        return buildings

    def is_inside_bounds(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def building_at(self, x: float, y: float, pad: float = 0.0) -> Rect | None:
        for b in self.buildings:
            if b.expanded(pad).contains(x, y):
                return b
        return None

    def is_road(self, x: float, y: float, pad: float = 0.0) -> bool:
        return self.is_inside_bounds(x, y) and self.building_at(x, y, pad) is None

    def random_road_point(self) -> tuple[float, float]:
        if self.rng.random() < 0.5:
            x = float(self.rng.choice(self.vertical_roads))
            y = float(self.rng.uniform(self.top + self.road * 0.65, self.bottom - self.road * 0.65))
        else:
            x = float(self.rng.uniform(self.left + self.road * 0.65, self.right - self.road * 0.65))
            y = float(self.rng.choice(self.horizontal_roads))
        return x, y

    def random_road_point_away_from(
        self,
        threats: list[tuple[float, float]],
        min_dist: float = 120.0,
        tries: int = 48,
    ) -> tuple[float, float]:
        best = self.random_road_point()
        best_score = -1.0
        for _ in range(tries):
            p = self.random_road_point()
            if not threats:
                return p
            d = min(float(np.hypot(p[0] - x, p[1] - y)) for x, y in threats)
            if d > best_score:
                best, best_score = p, d
            if d >= min_dist:
                return p
        return best

    def snap_to_road(self, x: float, y: float) -> tuple[float, float]:
        x = float(np.clip(x, self.left + self.road * 0.5, self.right - self.road * 0.5))
        y = float(np.clip(y, self.top + self.road * 0.5, self.bottom - self.road * 0.5))
        if self.is_road(x, y):
            return x, y

        vx = min(self.vertical_roads, key=lambda rx: abs(rx - x))
        hy = min(self.horizontal_roads, key=lambda ry: abs(ry - y))
        if abs(vx - x) < abs(hy - y):
            return float(vx), y
        return x, float(hy)

    def raycast_buildings(
        self,
        x: float,
        y: float,
        yaw: float,
        angles: np.ndarray,
        max_range: float = 90.0,
        step: float = 2.5,
    ) -> np.ndarray:
        """Cheap city raycasts for the evader driver module."""
        out = []
        for a in angles:
            heading = yaw + float(a)
            dx, dy = float(np.cos(heading)), float(np.sin(heading))
            dist = max_range
            t = 0.0
            while t <= max_range:
                px, py = x + dx * t, y + dy * t
                if not self.is_inside_bounds(px, py) or self.building_at(px, py, pad=1.0):
                    dist = t
                    break
                t += step
            out.append(dist)
        return np.asarray(out, dtype=np.float32)

    def resolve_vehicle(self, veh: Vehicle, radius: float = 2.35) -> float:
        """Push a vehicle out of buildings/world bounds and damp impact speed.

        Returns a rough impact magnitude used by reward and camera shake.
        """
        impact = 0.0
        b = self.building_at(veh.x, veh.y, pad=radius)
        if b is not None:
            ex = b.expanded(radius)
            eps = 1e-6
            options = [
                (abs(veh.x - ex.x0), np.array([-1.0, 0.0]), ex.x0 - eps),
                (abs(ex.x1 - veh.x), np.array([1.0, 0.0]), ex.x1 + eps),
                (abs(veh.y - ex.y0), np.array([0.0, -1.0]), ex.y0 - eps),
                (abs(ex.y1 - veh.y), np.array([0.0, 1.0]), ex.y1 + eps),
            ]
            _, normal, coord = min(options, key=lambda item: item[0])
            if normal[0] != 0.0:
                veh.x = float(coord)
            else:
                veh.y = float(coord)
            impact = self._dampen_world_velocity(veh, normal, bounce=0.08)

        if veh.x < self.left + radius:
            veh.x = self.left + radius
            impact = max(impact, self._dampen_world_velocity(veh, np.array([1.0, 0.0]), 0.05))
        elif veh.x > self.right - radius:
            veh.x = self.right - radius
            impact = max(impact, self._dampen_world_velocity(veh, np.array([-1.0, 0.0]), 0.05))
        if veh.y < self.top + radius:
            veh.y = self.top + radius
            impact = max(impact, self._dampen_world_velocity(veh, np.array([0.0, 1.0]), 0.05))
        elif veh.y > self.bottom - radius:
            veh.y = self.bottom - radius
            impact = max(impact, self._dampen_world_velocity(veh, np.array([0.0, -1.0]), 0.05))
        return impact

    @staticmethod
    def _dampen_world_velocity(veh: Vehicle, normal: np.ndarray, bounce: float) -> float:
        cy, sy = np.cos(veh.yaw), np.sin(veh.yaw)
        vw = np.array([veh.vx * cy - veh.vy * sy, veh.vx * sy + veh.vy * cy])
        vn = float(np.dot(vw, normal))
        if vn < 0.0:
            vw = vw - (1.0 + bounce) * vn * normal
            vw *= 0.58
            veh.vx = float(vw[0] * cy + vw[1] * sy)
            veh.vy = float(-vw[0] * sy + vw[1] * cy)
            veh.r *= 0.45
            return abs(vn)
        veh.vx *= 0.80
        veh.vy *= 0.80
        veh.r *= 0.65
        return 0.0
