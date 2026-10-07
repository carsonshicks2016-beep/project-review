"""
World layer: a `Car` couples a Vehicle (physics) with a Brain (controller)
and tracks its own fitness; `Simulation` runs the whole field around a Track.
"""

from __future__ import annotations
import math
import numpy as np

from .config import Config
from .physics import Vehicle
from .track import Track, make_track


def observe(track: Track, vehicle: Vehicle, sensors, car_spec, idx_hint):
    """Single source of truth for the policy observation, shared by the GA
    (`Car`) and the PPO env so the two can never drift apart.

    Layout: [n_rays raycasts] + [9 proprioceptive] + [curvature preview].
    Returns (obs, rays, ray_offsets, nearest_idx)."""
    rays, offsets, idx = track.cast_rays(vehicle.x, vehicle.y, vehicle.yaw,
                                         sensors, hint=idx_hint)
    v, c = vehicle, car_spec
    extras = np.array([
        v.vx / 40.0,
        v.vy / 20.0,
        v.r / 3.0,
        v.slip_angle / 0.7,
        v.rpm / c.redline_rpm,
        v.gear / (len(c.gear_ratios) - 1),
        v.lat_g / 1.5,
        v.steer / c.max_steer_angle,
        float(np.clip(v.aligning_torque / 1000.0, -2.0, 2.0)),
    ])
    curve = track.curvature_preview(idx, sensors.curve_preview_dists,
                                    track.spec.min_corner_radius)
    obs = np.concatenate([rays, extras, curve])
    return obs, rays, offsets, idx


class Car:
    def __init__(self, brain, cfg: Config, color):
        self.cfg = cfg
        self.brain = brain
        self.color = color
        self.vehicle = Vehicle(cfg.car, cfg.sim)
        self.finished = False
        self.reset_on(None)

    def reset_on(self, track: Track | None, offset_along=0.0, offset_lat=0.0):
        start_arc = 0.0
        if track is not None:
            # place along the actual (curving) centreline, not a straight tangent,
            # so staggered grid rows stay on the track
            i = track.index_at_arc(offset_along)
            cx, cy = track.center[i]
            tx, ty = track.tangent[i]
            nx, ny = track.normal[i]
            x = cx + nx * offset_lat
            y = cy + ny * offset_lat
            yaw = math.atan2(ty, tx)
            self.vehicle.reset(x, y, yaw)
            start_arc, self._idx, _ = track.progress(x, y, i)
        else:
            self._idx = 0
        self.alive = True
        self.finished = False
        self.fitness = 0.0
        self.start_distance = start_arc
        self.distance = start_arc    # signed metres along centreline (with laps)
        self.best_distance = 0.0     # best *travelled* distance from the start point
        self.laps = 0
        self._prev_arc = start_arc
        self._idle_t = 0.0
        self.time_alive = 0.0
        # last-frame snapshots (for dashboard / brain viz)
        self.rays = np.ones(self.cfg.sensors.n_rays)
        self.ray_offsets = np.zeros(self.cfg.sensors.n_rays)
        self.inputs = np.zeros(self.cfg.sensors.n_inputs)
        self.controls = (0.0, 0.0, 0.0)   # steer, throttle, brake (dash compat)
        self.clutch = 1.0
        self.shift_up = self.shift_down = False
        self.lateral = 0.0
        self.road_difficulty = 0.0
        # recurrent brains carry hidden state -> clear it for a fresh episode
        if hasattr(self.brain, "reset"):
            self.brain.reset()

    # ------------------------------------------------------------------ #
    def _build_inputs(self, track: Track):
        obs, self.rays, self.ray_offsets, self._idx = observe(
            track, self.vehicle, self.cfg.sensors, self.cfg.car, self._idx)
        self.inputs = obs
        return self.inputs

    def update(self, track: Track, dt: float):
        if not self.alive:
            return
        cfg = self.cfg
        x = self._build_inputs(track)
        out = self.brain.forward(x)
        steer, throttle, brake, clutch, su, sd = out
        self.controls = (steer, throttle, brake)
        self.clutch = clutch
        self.shift_up = su > 0.5
        self.shift_down = sd > 0.5

        sub = cfg.sim.physics_substeps
        ds = dt / sub
        for _ in range(sub):
            self.vehicle.step(throttle, brake, steer, ds,
                              clutch_cmd=clutch, shift_up=self.shift_up,
                              shift_down=self.shift_down,
                              manual_gears=cfg.control.manual_gears,
                              manual_clutch=cfg.control.manual_clutch)

        # progress along the track (handle lap wrap-around)
        arc, idx, lateral = track.progress(self.vehicle.x, self.vehicle.y, self._idx)
        self._idx = idx
        self.lateral = lateral
        self.road_difficulty = track.upcoming_difficulty(idx)
        if arc - self._prev_arc < -track.length * 0.5:
            self.laps += 1
        elif arc - self._prev_arc > track.length * 0.5:
            self.laps -= 1
        self._prev_arc = arc
        self.distance = self.laps * track.length + arc

        # Check if the car completed the required laps (finish condition)
        if self.laps >= cfg.evo.gen_max_laps:
            self.alive = False
            self.finished = True
            return

        self.time_alive += dt
        # fitness: distance *travelled* from the grid slot + small speed reward
        travelled = self.distance - self.start_distance
        self.fitness = travelled + 0.02 * self.vehicle.speed * self.time_alive

        # ---- death conditions ----
        if track.off_track(lateral):
            self.alive = False
            self.fitness -= 5.0          # discourage flying off
            return
        if travelled > self.best_distance + 0.5:
            self.best_distance = travelled
            self._idle_t = 0.0
        else:
            self._idle_t += dt
            if self._idle_t > cfg.evo.idle_kill_seconds:
                self.alive = False


class Simulation:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.track = make_track(cfg.track, kind="loop", seed=seed)
        self.cars: list[Car] = []
        self.generation = 0
        self.gen_time = 0.0
        self.focus = 0

    def new_track(self, seed=None, kind="loop"):
        self.track = make_track(self.cfg.track, kind=kind, seed=seed)

    def set_population(self, brains, colors):
        self.cars = [Car(b, self.cfg, colors[i % len(colors)])
                     for i, b in enumerate(brains)]
        self._grid_reset()
        self.gen_time = 0.0
        self._gen_best_dist = -1e9
        self._stagnant = 0.0
        self.focus = 0

    def _grid_reset(self):
        """Stagger cars on a starting grid so they don't spawn stacked."""
        cols = [-4.5, -1.5, 1.5, 4.5]
        for i, c in enumerate(self.cars):
            row = i // len(cols)
            c.reset_on(self.track, offset_along=3.0 + row * 5.5,
                       offset_lat=cols[i % len(cols)])

    def reset_episode(self):
        # counts as a new round/generation (used by PPO-live & watch modes,
        # where there's no evolve() step to advance the counter)
        self.generation += 1
        self._grid_reset()
        self.gen_time = 0.0
        self._gen_best_dist = -1e9
        self._stagnant = 0.0

    @property
    def alive_count(self):
        return sum(1 for c in self.cars if c.alive)

    @property
    def focus_car(self) -> Car:
        return self.cars[self.focus] if self.cars else None

    def cycle_focus(self, step=1):
        if self.cars:
            self.focus = (self.focus + step) % len(self.cars)

    def focus_leader(self):
        """Follow the best car that is still ALIVE (so the camera doesn't
        stick to a car that already crashed)."""
        if not self.cars:
            return
        alive = [i for i, c in enumerate(self.cars) if c.alive]
        pool = alive if alive else list(range(len(self.cars)))
        self.focus = max(pool, key=lambda i: self.cars[i].fitness)

    def step(self, dt: float) -> bool:
        """Advance one frame. The generation runs until everyone is dead/finished."""
        self.gen_time += dt
        for c in self.cars:
            c.update(self.track, dt)

        done = (self.alive_count == 0)
        return done
