"""Scripted driver policies used before learned MARL policies are trained."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .city import UrbanGrid
from .physics import Vehicle


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def world_velocity(veh: Vehicle) -> np.ndarray:
    cy, sy = np.cos(veh.yaw), np.sin(veh.yaw)
    return np.array([veh.vx * cy - veh.vy * sy, veh.vx * sy + veh.vy * cy])


def drive_toward(
    veh: Vehicle,
    target: tuple[float, float],
    desired_speed: float,
    handbrake_turns: bool = False,
    aggression: float = 1.0,
) -> tuple[float, float, float, float]:
    dx = float(target[0] - veh.x)
    dy = float(target[1] - veh.y)
    desired_yaw = float(np.arctan2(dy, dx))
    err = wrap_angle(desired_yaw - veh.yaw)
    steer = float(np.clip(err / 0.72, -1.0, 1.0))
    align = max(0.0, np.cos(err))
    speed_target = desired_speed * (0.28 + 0.72 * align)
    throttle = float(np.clip((speed_target - veh.speed) / 11.0, 0.0, 1.0))
    brake = float(np.clip((veh.speed - speed_target) / 13.0, 0.0, 1.0))
    handbrake = 0.0
    if handbrake_turns and abs(err) > 0.95 and veh.speed > 10.0:
        handbrake = float(np.clip((abs(err) - 0.8) / 1.0, 0.0, 1.0))
        throttle = max(throttle, 0.45)
        brake *= 0.35
    if abs(err) > 1.65 and veh.speed > 16.0:
        brake = max(brake, 0.35 * aggression)
        throttle *= 0.4
    return steer, throttle, brake, handbrake


@dataclass
class PursuerAssignment:
    role: str
    target: tuple[float, float]


class PursuerCoordinator:
    roles = ("front", "left", "right", "rear", "cut")

    def assignments(
        self,
        evader: Vehicle,
        pursuers: list[Vehicle],
        city: UrbanGrid,
        spoof_offsets: list[np.ndarray],
    ) -> list[PursuerAssignment]:
        ev_v = world_velocity(evader)
        heading = float(np.arctan2(ev_v[1], ev_v[0])) if np.linalg.norm(ev_v) > 1.0 else evader.yaw
        fwd = np.array([np.cos(heading), np.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])
        speed_lead = np.clip(evader.speed * 0.72, 10.0, 40.0)
        center = np.array([evader.x, evader.y]) + fwd * speed_lead
        offsets = {
            "front": fwd * 38.0,
            "left": left * 28.0 + fwd * 12.0,
            "right": -left * 28.0 + fwd * 12.0,
            "rear": -fwd * 32.0,
            "cut": fwd * 52.0 - left * 18.0,
        }
        out = []
        for i, _ in enumerate(pursuers):
            role = self.roles[i % len(self.roles)]
            target = center + offsets[role] + spoof_offsets[i]
            out.append(PursuerAssignment(role, city.snap_to_road(float(target[0]), float(target[1]))))
        return out


class EvaderPlanner:
    def __init__(self, city: UrbanGrid):
        self.city = city
        self.waypoint = city.random_road_point()
        self.last_axis: str | None = None

    def reset(self):
        self.waypoint = self.city.random_road_point()
        self.last_axis = None

    def choose_waypoint(self, evader: Vehicle, pursuers: list[Vehicle]):
        candidates = self._local_road_candidates(evader)
        if not candidates:
            threats = [(p.x, p.y) for p in pursuers]
            self.waypoint = self.city.random_road_point_away_from(threats, min_dist=140.0)
            self.last_axis = None
            return

        pos = np.array([evader.x, evader.y], dtype=float)
        velocity = world_velocity(evader)
        speed = float(np.linalg.norm(velocity))
        fwd = velocity / speed if speed > 1.0 else np.array([np.cos(evader.yaw), np.sin(evader.yaw)])
        threats = [np.array([p.x, p.y], dtype=float) for p in pursuers]

        best_score = -1e18
        best = candidates[0]
        for point, axis in candidates:
            vec = np.array(point, dtype=float) - pos
            dist = float(np.linalg.norm(vec))
            if dist < 42.0:
                continue
            direction = vec / max(dist, 1.0)
            alignment = float(np.dot(direction, fwd))
            target_dist_score = 1.0 - abs(dist - 74.0) / 74.0
            threat_dist = min((float(np.linalg.norm(np.array(point) - t)) for t in threats), default=180.0)
            escape_score = float(np.clip((threat_dist - 45.0) / 150.0, -1.0, 1.0))
            turn_bonus = 0.18 if self.last_axis is not None and axis != self.last_axis else 0.0
            score = 0.70 * target_dist_score + 0.45 * alignment + 0.85 * escape_score + turn_bonus
            if threat_dist < 38.0:
                score -= 2.0
            if score > best_score:
                best_score = score
                best = (point, axis)

        self.waypoint, self.last_axis = best

    def control(self, evader: Vehicle, pursuers: list[Vehicle]) -> tuple[float, float, float, float]:
        if np.hypot(evader.x - self.waypoint[0], evader.y - self.waypoint[1]) < 18.0:
            self.choose_waypoint(evader, pursuers)

        goal = np.array(self.waypoint, dtype=float)
        if pursuers:
            nearest = min(pursuers, key=lambda p: np.hypot(p.x - evader.x, p.y - evader.y))
            d = np.hypot(nearest.x - evader.x, nearest.y - evader.y)
            if d < 78.0:
                flee = np.array([evader.x - nearest.x, evader.y - nearest.y])
                flee_dir = flee / max(np.linalg.norm(flee), 1.0)
                if self.last_axis == "horizontal":
                    tangent = np.array([1.0, 0.0])
                    flee_dir = tangent * float(np.dot(flee_dir, tangent))
                elif self.last_axis == "vertical":
                    tangent = np.array([0.0, 1.0])
                    flee_dir = tangent * float(np.dot(flee_dir, tangent))
                goal += flee_dir * (82.0 - d)
        goal = np.array(self.city.snap_to_road(float(goal[0]), float(goal[1])))
        aim = self._road_aware_aimpoint(evader, (float(goal[0]), float(goal[1])))
        return drive_toward(evader, aim, 34.0, True, aggression=1.25)

    def _road_aware_aimpoint(self, evader: Vehicle, goal: tuple[float, float]) -> tuple[float, float]:
        """Return a short-horizon target that keeps fast slides inside road lanes."""
        x, y = float(evader.x), float(evader.y)
        gx, gy = self.city.snap_to_road(goal[0], goal[1])
        vx = min(self.city.vertical_roads, key=lambda rx: abs(rx - x))
        hy = min(self.city.horizontal_roads, key=lambda ry: abs(ry - y))
        gvx = min(self.city.vertical_roads, key=lambda rx: abs(rx - gx))
        ghy = min(self.city.horizontal_roads, key=lambda ry: abs(ry - gy))
        on_vertical = abs(x - vx) <= self.city.road * 0.62
        on_horizontal = abs(y - hy) <= self.city.road * 0.62

        target_axis = self.last_axis
        if target_axis not in {"horizontal", "vertical"}:
            target_axis = "vertical" if abs(gx - gvx) < abs(gy - ghy) else "horizontal"

        if target_axis == "horizontal":
            road_y = ghy
            if not on_horizontal and on_vertical:
                return self._axis_lookahead(x, y, road_y, vx, "vertical")
            return self._axis_lookahead(x, y, gx, road_y, "horizontal")

        road_x = gvx
        if not on_vertical and on_horizontal:
            return self._axis_lookahead(x, y, road_x, hy, "horizontal")
        return self._axis_lookahead(x, y, gy, road_x, "vertical")

    def _axis_lookahead(
        self,
        x: float,
        y: float,
        target_primary: float,
        road_center: float,
        axis: str,
    ) -> tuple[float, float]:
        delta = target_primary - (x if axis == "horizontal" else y)
        if abs(delta) < 18.0:
            if axis == "horizontal":
                return self.city.snap_to_road(target_primary, road_center)
            return self.city.snap_to_road(road_center, target_primary)

        sign = 1.0 if delta >= 0.0 else -1.0
        lookahead = float(np.clip(abs(delta), 26.0, 54.0))
        lane_error = road_center - (y if axis == "horizontal" else x)
        lane_bias = float(np.clip(lane_error * 1.35, -self.city.road * 0.34, self.city.road * 0.34))
        if axis == "horizontal":
            return self.city.snap_to_road(x + sign * lookahead, road_center + lane_bias)
        return self.city.snap_to_road(road_center + lane_bias, y + sign * lookahead)

    def _local_road_candidates(self, evader: Vehicle) -> list[tuple[tuple[float, float], str]]:
        x, y = float(evader.x), float(evader.y)
        vx = min(self.city.vertical_roads, key=lambda rx: abs(rx - x))
        hy = min(self.city.horizontal_roads, key=lambda ry: abs(ry - y))
        on_vertical = abs(x - vx) <= self.city.road * 0.72
        on_horizontal = abs(y - hy) <= self.city.road * 0.72
        distances = (54.0, 74.0, 96.0, 118.0)
        raw: list[tuple[tuple[float, float], str]] = []
        if on_vertical or not on_horizontal:
            for dist in distances:
                raw.append(((vx, y + dist), "vertical"))
                raw.append(((vx, y - dist), "vertical"))
        if on_horizontal or not on_vertical:
            for dist in distances:
                raw.append(((x + dist, hy), "horizontal"))
                raw.append(((x - dist, hy), "horizontal"))

        candidates: list[tuple[tuple[float, float], str]] = []
        seen: set[tuple[int, int, str]] = set()
        for point, axis in raw:
            px, py = self.city.snap_to_road(point[0], point[1])
            if not self.city.is_road(px, py, pad=1.0):
                continue
            key = (round(px), round(py), axis)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(((float(px), float(py)), axis))
        return candidates
