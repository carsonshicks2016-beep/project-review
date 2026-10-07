"""
Tire skid marks — dark rubber traces painted on the road while drifting.
Accumulate for the life of a generation and are cleared on reset.
"""
import math
import pygame

_CAR_H = 22   # mirrored from car.py — avoids circular import
_CAR_W = 12


class Skidmarks:
    MAX = 6000   # maximum stored line segments

    def __init__(self):
        self.marks = []   # list of (x1, y1, x2, y2)
        self._prev  = {}  # (car_id, side) → (x, y) of last point

    def update(self, cars):
        """Call once per sim step with the current car list."""
        for car in cars:
            cid = id(car)
            if not car.alive or not car.is_drifting:
                # Clear stored positions so we don't draw phantom lines
                # when the car starts drifting again later.
                self._prev.pop((cid, -1), None)
                self._prev.pop((cid,  1), None)
                continue

            fx = math.cos(car.angle)
            fy = math.sin(car.angle)
            lx, ly = -fy, fx

            # Centre of rear axle
            rx = car.x - fx * (_CAR_H * 0.5)
            ry = car.y - fy * (_CAR_H * 0.5)

            for side in (-1, 1):
                tx = rx + lx * (_CAR_W * 0.5) * side
                ty = ry + ly * (_CAR_W * 0.5) * side
                key  = (cid, side)
                prev = self._prev.get(key)
                if prev:
                    ddx = tx - prev[0]
                    ddy = ty - prev[1]
                    # Skip if the car was teleported by a collision impulse
                    if ddx * ddx + ddy * ddy < 80 ** 2:
                        self.marks.append((prev[0], prev[1], tx, ty))
                self._prev[key] = (tx, ty)

        # Trim oldest marks if over the cap
        if len(self.marks) > self.MAX:
            del self.marks[:len(self.marks) - self.MAX]

    def reset(self):
        self.marks.clear()
        self._prev.clear()

    def draw(self, surface, offset, screen_w, screen_h):
        if not self.marks:
            return
        ox, oy = offset
        col    = (26, 20, 12)   # dark rubber-brown
        M      = 40

        for x1, y1, x2, y2 in self.marks:
            sx1 = int(x1 - ox)
            sy1 = int(y1 - oy)
            # Quick viewport cull — check only the start point
            if not (-M < sx1 < screen_w + M and -M < sy1 < screen_h + M):
                continue
            pygame.draw.line(surface, col,
                             (sx1, sy1),
                             (int(x2 - ox), int(y2 - oy)), 2)
