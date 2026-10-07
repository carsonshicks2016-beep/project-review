from __future__ import annotations

import math


TAU = math.tau


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += TAU
    while angle > math.pi:
        angle -= TAU
    return angle


def angle_to(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(y1 - y0, x1 - x0)


def angle_delta(target: float, current: float) -> float:
    return wrap_angle(target - current)
