"""
CarAgent — one brain driving one car, with fitness + crash detection.

Shared by headless GA evaluation and the live population view, so both score
cars identically. The brain decides at `control_hz`; the physics runs at full
rate in between. An automatic gearbox handles clutch + shifting.

Fitness = distance covered along the centreline (laps x track length), with a
small movement term so even gen-0 cars get a usable gradient. A run ends on a
crash (off-track too long / way off line), a stall, a long stretch with no
progress, going backwards, or the time budget (truncation).
"""
from __future__ import annotations

import numpy as np

from .physics import Controls, Vehicle


class CarAgent:
    def __init__(self, brain, spec, sim, trk, sensors, evo, autobox_cls,
                 color=(220, 90, 90), index=0):
        self.brain = brain
        self.spec = spec
        self.sim = sim
        self.trk = trk
        self.sensors = sensors
        self.evo = evo
        self.color = color
        self.index = index
        self.control_period = max(1, round(1.0 / (evo.control_hz * sim.dt)))
        self.veh = Vehicle(spec, sim)
        self.box = autobox_cls(spec)
        self.reset()

    def reset(self):
        sx, sy, syaw = self.trk.start_pose()
        self.veh.reset(sx, sy, syaw)
        self.box.__init__(self.spec)
        self.done = False
        self.fitness = 0.0
        self.act = np.zeros(3)
        self.obs = None
        self._ctr = 0
        # fitness / progress bookkeeping
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.prev_frac = fr["progress"]
        self.cum = 0.0
        self.max_cum = 0.0
        self.best_cum = 0.0
        self.total_dist = 0.0
        self.time = 0.0
        self.off_t = 0.0
        self.stall_t = 0.0
        self.since_prog = 0.0
        self.trail = [(self.veh.x, self.veh.y)]
        return self

    @property
    def laps(self) -> float:
        return self.max_cum

    # ------------------------------------------------------------------ #
    def step(self, dt: float):
        if self.done:
            return
        if self._ctr % self.control_period == 0:
            self.obs = self.sensors.observe(self.veh, self.trk)
            self.act = self.brain.forward(self.obs.vector)
        self._ctr += 1

        steer, thr, brk = float(self.act[0]), float(self.act[1]), float(self.act[2])
        clutch, up, down = self.box.update(self.veh, thr, dt)
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.surface_grip = self.spec.offtrack_grip if fr["off_track"] else 1.0
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"],
                          fr["vcurv"])
        self.veh.step(Controls(steer=steer, throttle=thr, brake=brk,
                               clutch=clutch, shift_up=up, shift_down=down))
        self._update(fr, dt)

    def _update(self, fr, dt):
        self.time += dt
        self.total_dist += self.veh.speed * dt
        if len(self.trail) == 0 or self._ctr % 6 == 0:
            self.trail.append((self.veh.x, self.veh.y))

        # cumulative progress with lap wrap
        frac = fr["progress"]
        d = frac - self.prev_frac
        if d < -0.5:
            d += 1.0
        elif d > 0.5:
            d -= 1.0
        self.cum += d
        self.prev_frac = frac
        self.max_cum = max(self.max_cum, self.cum)

        # timers
        self.off_t = self.off_t + dt if fr["off_track"] else 0.0
        crawling = self.time > self.evo.start_grace and self.veh.speed < 0.8
        self.stall_t = self.stall_t + dt if crawling else 0.0
        if self.cum > self.best_cum + 1e-3:
            self.best_cum = self.cum
            self.since_prog = 0.0
        else:
            self.since_prog += dt

        # termination
        crashed = (self.off_t > self.evo.offtrack_timeout
                   or abs(fr["lateral"]) > self.trk.half + 4.0)
        stalled = self.stall_t > self.evo.stall_timeout
        no_prog = self.since_prog > self.evo.progress_timeout
        backward = self.cum < self.max_cum - 0.04
        timeout = self.time > self.evo.episode_seconds
        if crashed or stalled or no_prog or backward or timeout:
            self._finish()

    def _finish(self):
        self.done = True
        self.fitness = self.max_cum * self.trk.length + 0.02 * self.total_dist

    # ------------------------------------------------------------------ #
    def run(self, max_seconds: float | None = None) -> float:
        """Simulate to completion (headless) and return fitness."""
        budget = max_seconds if max_seconds is not None else self.evo.episode_seconds + 1
        max_steps = int(budget / self.sim.dt)
        for _ in range(max_steps):
            if self.done:
                break
            self.step(self.sim.dt)
        if not self.done:
            self._finish()
        return self.fitness
