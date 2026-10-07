"""Single-track vehicle model with a combined-slip tyre, integrated at 400 Hz.

The point of this file is that drifting is never scripted: the rear tyre has a
peak and falls off past it, so lifting, stabbing the throttle or yanking the
handbrake breaks traction the way it does in a real car, and the only way back
is opposite lock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from drift.config import CarConfig, TireParams

G = 9.81
_EPS = 1e-6


def magic(slip: float, tp: TireParams) -> float:
    """Normalised magic formula: peaks at ~1.0 then falls away."""
    bs = tp.B * slip
    return math.sin(tp.C * math.atan(bs - tp.E * (bs - math.atan(bs))))


def tire_forces(alpha: float, kappa: float, fz: float, tp: TireParams) -> tuple[float, float]:
    """Combined longitudinal/lateral force from one axle via the slip circle.

    ``alpha`` is the slip angle (rad), ``kappa`` the longitudinal slip ratio.
    Returns (Fx, Fy) in the wheel frame; Fy opposes a positive slip angle.
    """
    if fz <= 0.0:
        return 0.0, 0.0
    sx = kappa
    sy = math.tan(max(-1.5, min(1.5, alpha)))
    s = math.hypot(sx, sy)
    if s < _EPS:
        return 0.0, 0.0
    force = tp.mu * fz * magic(s, tp)
    return force * sx / s, -force * sy / s


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    u: float = 0.0            # body-longitudinal velocity, m/s
    v: float = 0.0            # body-lateral velocity, m/s
    r: float = 0.0            # yaw rate, rad/s
    omega_r: float = 0.0      # rear wheel speed, rad/s
    steer: float = 0.0        # actual roadwheel angle, rad
    ax: float = 0.0           # longitudinal accel, kept for load transfer

    @property
    def speed(self) -> float:
        return math.hypot(self.u, self.v)

    @property
    def slip_rear(self) -> float:
        """Rear-axle slip angle: the number the whole scoring system is about."""
        return math.atan2(self.v - 1.45 * self.r, max(abs(self.u), 2.0)) * (
            1.0 if self.u >= 0 else -1.0
        )

    @property
    def beta(self) -> float:
        """Chassis sideslip: angle between where it points and where it goes."""
        return math.atan2(self.v, max(abs(self.u), 0.5))


class CarModel:
    def __init__(self, cfg: CarConfig | None = None) -> None:
        self.cfg = cfg or CarConfig()
        self.L = self.cfg.a + self.cfg.b

    # -- slip angles -------------------------------------------------------
    def slip_angles(self, s: CarState) -> tuple[float, float]:
        c = self.cfg
        ux = max(abs(s.u), 2.0)          # low-speed guard; the model is singular at 0
        af = math.atan2(s.v + c.a * s.r, ux) - s.steer
        ar = math.atan2(s.v - c.b * s.r, ux)
        return af, ar

    def slip_ratio(self, s: CarState) -> float:
        c = self.cfg
        ux = max(abs(s.u), 2.0)
        return (s.omega_r * c.wheel_radius - s.u) / ux

    # -- one control step --------------------------------------------------
    def step(self, s: CarState, steer_cmd: float, throttle: float, handbrake: float) -> CarState:
        """Advance ``s`` by one control interval. Commands are in [-1, 1] / [0, 1]."""
        c = self.cfg
        h = c.dt / c.substeps
        steer_target = max(-1.0, min(1.0, steer_cmd)) * c.max_steer
        throttle = max(-1.0, min(1.0, throttle))
        handbrake = max(0.0, min(1.0, handbrake))

        for _ in range(c.substeps):
            # Steering actuator: rate limited, so you cannot teleport to lock.
            d = steer_target - s.steer
            max_d = c.steer_rate * h
            s.steer += max(-max_d, min(max_d, d))

            # Static load plus longitudinal transfer (squat under power). The
            # transfer follows the *specific force* an accelerometer would read,
            # i.e. Fx/m alone: folding in the v*r frame term instead makes load
            # feed its own tyre force and the model diverges. Clamped as well,
            # since nothing physical lets one axle carry twice the car.
            transfer = c.mass * s.ax * c.cg_height / self.L
            lim = 0.75 * c.mass * G
            transfer = max(-lim, min(lim, transfer))
            fz_f = max(0.0, c.mass * G * c.b / self.L - transfer)
            fz_r = max(0.0, c.mass * G * c.a / self.L + transfer)

            af, ar = self.slip_angles(s)
            kappa = self.slip_ratio(s)

            fxf, fyf = tire_forces(af, 0.0, fz_f, c.front_tire)
            fxr, fyr = tire_forces(ar, kappa, fz_r, c.rear_tire)

            # Driveline: torque falls off as a power limit once you are rolling.
            wheel_w = max(abs(s.omega_r), 1.0)
            t_cap = min(c.drive_torque, c.drive_power / (wheel_w * c.wheel_radius) * c.wheel_radius)
            if throttle >= 0.0:
                t_drive = throttle * t_cap
            else:
                t_drive = throttle * c.brake_torque
            t_hand = -handbrake * c.handbrake_torque * math.copysign(1.0, s.omega_r) if abs(s.omega_r) > 0.5 else 0.0

            # Rear wheel spin dynamics: this is what lets the throttle break grip.
            domega = (t_drive + t_hand - fxr * c.wheel_radius) / c.wheel_inertia
            s.omega_r += domega * h
            if handbrake > 0.5 and abs(s.omega_r) < 4.0:
                s.omega_r = 0.0          # locked rear axle

            drag = c.drag_coef * s.u * abs(s.u) + c.roll_resist * c.mass * G * math.copysign(1.0, s.u)
            sd, cd = math.sin(s.steer), math.cos(s.steer)

            fx = fxf * cd - fyf * sd + fxr - drag
            fy = fxf * sd + fyf * cd + fyr
            mz = c.a * (fxf * sd + fyf * cd) - c.b * fyr

            du = fx / c.mass + s.v * s.r
            dv = fy / c.mass - s.u * s.r
            dr = mz / c.inertia

            s.ax = fx / c.mass          # specific force, for the next step's load transfer
            s.u += du * h
            s.v += dv * h
            s.r += dr * h

            # Kill the residual crawl so a parked car actually stops.
            if abs(s.u) < 0.15 and abs(throttle) < 0.05:
                s.u *= 0.85
                s.v *= 0.85

            s.x += (s.u * math.cos(s.yaw) - s.v * math.sin(s.yaw)) * h
            s.y += (s.u * math.sin(s.yaw) + s.v * math.cos(s.yaw)) * h
            s.yaw = (s.yaw + s.r * h + math.pi) % (2 * math.pi) - math.pi

        return s

    def spawn(self, x: float, y: float, yaw: float, speed: float) -> CarState:
        s = CarState(x=x, y=y, yaw=yaw, u=speed, v=0.0, r=0.0)
        s.omega_r = speed / self.cfg.wheel_radius
        return s
