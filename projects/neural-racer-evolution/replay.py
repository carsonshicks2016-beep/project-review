"""
Hall-of-fame replay mode.
Loads saved brains and races them on the current track with no evolution.
Press H in-game to enter; press H again (or ESC) to return to training.
"""

import pygame
import math
import hall_of_fame as hof
from car import Car


GHOST_ALPHA = 200   # opacity of replay cars (0–255)


def _tint(color, alpha_frac):
    """Return a dimmed version of color for ghost rendering."""
    return tuple(int(c * alpha_frac) for c in color)


class ReplaySession:
    """
    Runs a set of hall-of-fame brains on a given track.
    Call update(dt) each frame, draw(surface, offset) to render.
    """

    # Palette for the replay cars — cycles if more than len(PALETTE) brains
    PALETTE = [
        (255, 220,  60),   # gold   — best brain
        (180, 180, 255),   # soft blue
        (100, 255, 180),   # mint
        (255, 130, 100),   # salmon
        (200,  80, 255),   # purple
        (80,  200, 255),   # cyan
        (255, 255, 255),   # white
        (255, 180,  80),   # amber
        (120, 255, 120),   # lime
        (255,  80, 160),   # pink
        (160, 220, 255),   # sky
        (255, 160,  40),   # orange
    ]

    def __init__(self, track):
        self.track    = track
        self.cars     = []
        self.labels   = []   # short string label per car
        self.sim_time = 0.0
        self._load()

    def _load(self):
        saved = hof.list_all()
        if not saved:
            return

        sx, sy, sa = self.track.get_start_pos()

        for i, path in enumerate(saved):
            try:
                genome, gen, fitness = hof.load(path)
            except Exception:
                continue

            # Spread replay cars laterally at start so they don't all overlap
            import math as _m
            lx = -_m.sin(sa)
            ly =  _m.cos(sa)
            n   = len(saved)
            lat = (i - (n - 1) / 2.0) * 20.0   # 20 px spacing
            car = Car(sx + lx * lat, sy + ly * lat, sa, genome)
            car.color = self.PALETTE[i % len(self.PALETTE)]
            self.cars.append(car)

            label = f"G{gen} {int(fitness):,}"
            self.labels.append(label)

    def update(self, dt):
        self.sim_time += dt
        for car in self.cars:
            if car.alive:
                car.update(dt, self.track)

    def draw(self, surface, offset, font):
        ox, oy = offset

        # Draw all cars
        for i, car in enumerate(self.cars):
            car.draw(surface, offset, draw_rays=False)

            # Floating label above each car
            if car.alive:
                lx = int(car.x - ox)
                ly = int(car.y - oy) - 20
                txt = font.render(self.labels[i], True, car.color)
                surface.blit(txt, (lx - txt.get_width() // 2, ly))

    def best_alive(self):
        alive = [c for c in self.cars if c.alive]
        if alive:
            return max(alive, key=lambda c: c.fitness)
        return max(self.cars, key=lambda c: c.fitness) if self.cars else None

    @property
    def alive_count(self):
        return sum(1 for c in self.cars if c.alive)

    @property
    def loaded_count(self):
        return len(self.cars)
