"""
Speed heatmap — records where cars travel and how fast,
then paints the track surface with a colour gradient:
  cool blue  →  white  →  hot orange/red
Brighter = faster.  Resets each generation.
"""

import numpy as np
import pygame


# How many world-pixels each heatmap cell covers.
# Smaller = finer detail (racing-line resolution) but more memory & slower to render.
# Car is 22×12 px; CELL=8 traces the actual racing line at sub-car resolution.
CELL = 8


class Heatmap:
    def __init__(self, bbox):
        """
        bbox: (xmin, ymin, xmax, ymax) — world-space bounds of the track.
        """
        xmin, ymin, xmax, ymax = bbox
        self.xmin = xmin
        self.ymin = ymin
        self.cols = int((xmax - xmin) / CELL) + 2
        self.rows = int((ymax - ymin) / CELL) + 2

        self._speed_sum   = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._visit_count = np.zeros((self.rows, self.cols), dtype=np.float32)

        self._surface         = None   # cached pygame Surface, rebuilt when dirty
        self._dirty           = True
        self._records_since   = 0      # rate-limiter — rebuild every N records
        self._rebuild_every   = 15     # at 60 Hz → ~4 rebuilds/sec

    # ------------------------------------------------------------------
    # Data recording
    # ------------------------------------------------------------------

    def record(self, cars):
        """Call once per sim step with the current live car list."""
        for car in cars:
            if not car.alive:
                continue
            c = int((car.x - self.xmin) / CELL)
            r = int((car.y - self.ymin) / CELL)
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self._speed_sum[r, c]   += car.speed
                self._visit_count[r, c] += 1.0
        # Rate-limit surface rebuilds — data updates every frame, surface every ~250ms
        self._records_since += 1
        if self._records_since >= self._rebuild_every:
            self._dirty         = True
            self._records_since = 0

    def reset(self):
        self._speed_sum[:]   = 0.0
        self._visit_count[:] = 0.0
        self._dirty = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rebuild_surface(self, max_speed):
        """Render cell grid into an RGBA surface scaled by average speed.

        Builds a tiny per-cell RGBA numpy buffer (cols × rows × 4) directly,
        then scales it up to world dimensions with pygame.transform.scale
        (nearest-neighbor — keeps cells sharp). Dramatically faster than
        looping pygame.draw.rect once per cell.
        """
        if max_speed < 1.0:
            max_speed = 1.0

        visited = self._visit_count > 0
        avg     = np.where(visited,
                           self._speed_sum / np.maximum(self._visit_count, 1),
                           0.0)
        t       = np.clip(avg / max_speed, 0.0, 1.0)

        # Vectorised blue → white → orange → red gradient
        low     = t < 0.5
        tt_low  = t * 2.0
        tt_high = (t - 0.5) * 2.0
        R = np.where(low, 40  + tt_low * 215.0,            255.0)
        G = np.where(low, 100 + tt_low * 155.0,            255.0 - tt_high * 200.0)
        B = np.where(low, 220 - tt_low * 220.0,            0.0)
        A = np.where(visited,
                     np.clip(30.0 + np.minimum(self._visit_count, 60) * 2.5, 0, 180),
                     0.0)

        # Pack into (rows, cols, 4) uint8 buffer
        rgba = np.stack([R, G, B, A], axis=-1).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)

        # pygame.image.frombuffer interprets bytes as (height=rows, width=cols)
        small = pygame.image.frombuffer(rgba.tobytes(),
                                        (self.cols, self.rows), 'RGBA')

        # Scale up (nearest-neighbor by default — preserves sharp cell edges)
        self._surface = pygame.transform.scale(
            small, (self.cols * CELL, self.rows * CELL))
        self._dirty   = False

    def draw(self, screen, offset, max_speed=540.0):
        """Blit the heatmap onto screen, accounting for camera offset."""
        if self._dirty or self._surface is None:
            self._rebuild_surface(max_speed)

        ox, oy = offset
        blit_x = int(self.xmin - ox)
        blit_y = int(self.ymin - oy)
        screen.blit(self._surface, (blit_x, blit_y))
