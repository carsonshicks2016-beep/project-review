"""The RallyAI side of the vendored dynamics.

Everything here is *driver and world coupling*, not vehicle physics:

* feeding the road plane and surface friction to the model each step,
* an automatic gearbox, because the agent's action space is four continuous
  inputs and shifting is not one of them,
* translating a policy action into ``Controls``,
* resolving collisions with obstacles and the world.

The vendored model already works in RallyAI's coordinate system — ``x``/``y`` on
the ground, ``z`` up, ``yaw`` about ``+z``, steering positive to the left — so
this is genuinely a thin binding and not a conversion layer. That was the point
of choosing z-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rallyai.track import Track, TrackQuery

from .cars import evo_rally, surface_grip
from .vendor import CarSpec, Controls, SimSpec, Vehicle

# Impact speed above which a collision is a crash rather than a scrape. Below
# it, the car is pushed out and loses speed; above it, the episode is over.
CRASH_SPEED = 8.0

# Extra grip penalty for the surface beyond the corridor edge. Multiplies the
# spec's own offtrack_grip. Leaving the road should cost time, not instantly
# end the run — that is what the off-course timer is for.
OFF_TRACK_GRIP_SCALE = 1.0


@dataclass(slots=True)
class Impact:
    """A collision resolved this step."""

    speed: float          # closing speed along the contact normal, m/s
    depth: float          # penetration, metres
    crash: bool           # hard enough to end the episode


class AutoBox:
    """An automatic gearbox.

    The agent drives with ``[steer, throttle, brake, handbrake]``; shifting is
    not in the action space, so something has to choose gears. A real rally
    driver does shift, so this is a simplification — but it is a *driver* model,
    not a physics change, and keeping it here means the policy never learns to
    exploit gearbox quirks it cannot see.

    Ported from Supra Ai 2's ``AutoBox``, where it lived in the rendering module.
    Retuned for a close-ratio dogbox: quicker shifts, and an upshift point just
    under the limiter.
    """

    def __init__(self, spec: CarSpec, shift_time: float = 0.18):
        self.spec = spec
        self.shift_time = shift_time
        self.cooldown = 0.0
        self.clutch = 0.0     # start disengaged: no creep on the start line

    def reset(self) -> None:
        self.cooldown = 0.0
        self.clutch = 0.0

    def update(self, veh: Vehicle, throttle: float, dt: float) -> tuple[float, bool, bool]:
        """Return ``(clutch, shift_up, shift_down)`` for this step."""
        self.cooldown = max(0.0, self.cooldown - dt)
        up = down = False
        rpm = veh.rpm

        if self.cooldown == 0.0:
            if rpm > self.spec.redline_rpm * 0.96 and veh.gear < len(self.spec.gear_ratios):
                up = True
                self.cooldown = self.shift_time
            elif rpm < self.spec.idle_rpm * 1.9 and veh.gear > 1 and veh.speed > 1.0:
                down = True
                self.cooldown = self.shift_time

        # Dip the clutch through a shift. Disengage when stopped or coasting
        # down to idle, so the idle governor does not fight the brakes; engaged
        # otherwise, so lift-off engine braking still rotates the car.
        idle_thresh = self.spec.idle_rpm * 1.25
        if up or down:
            target = 0.2
        elif throttle < 0.05 and (veh.speed < 1.5 or rpm < idle_thresh):
            target = 0.0
        else:
            target = 1.0
        self.clutch += (target - self.clutch) * min(1.0, dt * 12.0)
        return self.clutch, up, down


class RallyCar:
    """A car bound to a stage.

    Owns the vendored ``Vehicle``, the gearbox, and the per-step coupling to the
    road. Knows nothing about reward, observation or episodes — that is the
    environment's job.
    """

    def __init__(self, spec: CarSpec | None = None, sim: SimSpec | None = None):
        self.spec = spec or evo_rally()
        self.sim = sim or SimSpec()
        self.vehicle = Vehicle(self.spec, self.sim)
        self.box = AutoBox(self.spec)
        self.last_impact: Impact | None = None
        self._was_airborne = False
        self.took_off = False
        self.landed = False
        # Attitude at the instant of touchdown. The vendored model overwrites
        # ``pitch`` with the road grade in the same step it lands, so the
        # airborne attitude — whether the car came in nose-first or flat — is
        # gone by the time anything downstream can read it. Captured here
        # because judging a landing is impossible without it.
        self.landing_pitch = 0.0
        self.landing_slip = 0.0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def place_at_start(self, track: Track, speed: float = 0.0) -> TrackQuery:
        """Put the car on the start line, aligned with the road and sitting on it."""
        s = track.start_s
        i = int(np.clip(round(s / track.ds), 0, len(track.s) - 1))
        x, y = float(track.x[i]), float(track.y[i])
        yaw = float(track.start_heading)

        self.vehicle.reset(x=x, y=y, yaw=yaw, speed=speed)
        self.box.reset()
        self.last_impact = None
        self._was_airborne = False
        self.took_off = self.landed = False
        self.landing_pitch = self.landing_slip = 0.0

        query = track.project(x, y, yaw=yaw, hint_s=s)
        # Seed the road plane before the first step, or the opening frame is
        # computed against flat ground and the car visibly settles.
        self.apply_road(query)
        self.vehicle.z = query.z
        self.vehicle.road_z = query.z
        return query

    # ------------------------------------------------------------------ #
    # per-step coupling
    # ------------------------------------------------------------------ #

    def apply_road(self, query: TrackQuery) -> None:
        """Hand the model the road plane and the friction under the tyres."""
        grip = surface_grip(query.mu)
        if not query.on_track:
            grip *= self.spec.offtrack_grip * OFF_TRACK_GRIP_SCALE
        self.vehicle.surface_grip = grip
        self.vehicle.set_road(
            grade=query.grade,
            # Negated: the two sides of this call use OPPOSITE sign conventions.
            # The contract's ``camber`` is positive when the RIGHT edge is
            # raised, which is what banks into a left-hander -- and geometry.py
            # agrees, computing road height as ``z + lat * sin(camber)``. The
            # vendored model's ``bank`` is positive when the LEFT side is up
            # (``physics.py:107``), so passing camber through unchanged made
            # gravity push toward the raised edge.
            #
            # The effect was silent and backwards: the generator banks every
            # corner into the turn, so every banked corner was played OFF-camber
            # by exactly the amount intended to help. Measured on a 45 m
            # left-hander at 1.05 g -- worst lateral excursion 2.57 m with the
            # sign as shipped against 1.60 m with it corrected.
            bank=-query.camber,
            heading=query.heading,
            z=query.z,
            vcurv=query.vcurv,
        )

    def step(self, action, dt: float) -> None:
        """Advance one fixed step.

        ``action`` is ``[steer, throttle, brake, handbrake]``, each already in
        range. Steering is rate-limited inside the model, not clamped here — the
        policy commands a target and the car takes time to get there.
        """
        steer, throttle, brake, handbrake = (float(a) for a in action)
        clutch, up, down = self.box.update(self.vehicle, throttle, dt)

        was_air = self.vehicle.airborne
        # Read before the step: if this is the step that lands, the model resets
        # pitch to the road grade before returning.
        pitch_before = self.vehicle.pitch
        slip_before = self.vehicle.slip_angle
        self.vehicle.step(
            Controls(
                steer=steer,
                throttle=throttle,
                brake=brake,
                clutch=clutch,
                handbrake=handbrake,
                shift_up=up,
                shift_down=down,
            ),
            dt,
        )
        now_air = self.vehicle.airborne
        self.took_off = now_air and not was_air
        self.landed = was_air and not now_air
        self._was_airborne = now_air
        if self.landed:
            self.landing_pitch = pitch_before
            self.landing_slip = slip_before

    # ------------------------------------------------------------------ #
    # collision
    # ------------------------------------------------------------------ #

    def resolve_collision(self, track: Track, query: TrackQuery) -> Impact | None:
        """Push the car out of any obstacle it has entered, and report the hit.

        Only obstacles are solid. The corridor edge is *not* a wall — a rally car
        can run wide onto the verge and come back, which is what the off-course
        timer is for. The agent sees both through raycast; only one of them
        stops the car.
        """
        v = self.vehicle
        radius = 0.5 * max(self.spec.body_width, 1.0)
        normal, depth = track.collide_obstacles(v.x, v.y, radius, hint_s=query.s)
        if normal is None:
            self.last_impact = None
            return None

        # Closing speed along the contact normal, in world axes.
        world_vx = v.vx * np.cos(v.yaw) - v.vy * np.sin(v.yaw)
        world_vy = v.vx * np.sin(v.yaw) + v.vy * np.cos(v.yaw)
        closing = -(world_vx * normal[0] + world_vy * normal[1])
        closing = max(0.0, float(closing))

        # Push out of penetration.
        v.x += float(normal[0]) * depth
        v.y += float(normal[1]) * depth

        # Kill the component of velocity into the obstacle and scrub the rest.
        if closing > 0.0:
            body_nx = normal[0] * np.cos(v.yaw) + normal[1] * np.sin(v.yaw)
            body_ny = -normal[0] * np.sin(v.yaw) + normal[1] * np.cos(v.yaw)
            v_into = v.vx * body_nx + v.vy * body_ny
            if v_into < 0.0:
                v.vx -= v_into * body_nx
                v.vy -= v_into * body_ny
            v.vx *= 0.55
            v.vy *= 0.55
            v.r *= 0.5

        impact = Impact(speed=closing, depth=float(depth), crash=closing >= CRASH_SPEED)
        self.last_impact = impact
        return impact

    # ------------------------------------------------------------------ #
    # readouts
    # ------------------------------------------------------------------ #

    @property
    def speed(self) -> float:
        return float(self.vehicle.speed)

    def mean_grip_used(self) -> float:
        """Mean fraction of the friction budget in use across the four tyres."""
        return float(np.mean(self.vehicle.wheel_grip))
