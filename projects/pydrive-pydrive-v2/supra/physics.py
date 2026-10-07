"""
Four-wheel vehicle dynamics.

A proper four-corner model, not a bicycle approximation:

  * each wheel has its own contact patch with slip angle + slip ratio,
  * a single combined-slip Pacejka tyre model (slip-vector normalisation), so
    lateral and longitudinal demands share one friction budget correctly,
  * tyre forces build over a relaxation length (slip lag), giving real
    transient behaviour instead of instantaneous grip,
  * vertical loads carry dynamic longitudinal + (roll-stiffness-distributed)
    lateral weight transfer, plus speed-dependent downforce, settled through a
    first-order suspension lag with load-sensitive peak friction,
  * a sub-stepped drivetrain links engine flywheel -> clutch -> gearbox ->
    LSD -> driven wheels, with sequential twin-turbo spool,
  * aero drag opposes the full velocity vector (so big slip angles scrub
    speed), plus rolling resistance and an off-track grip multiplier,
  * 2.5D road-plane forces (PHYSICS_3D_PLAN): the track feeds set_road() each
    step; slope gravity becomes a real body force, normal load scales with
    slope and v^2 crest/dip curvature, pitch/roll are kinematic outputs.
    Callers that never call set_road() get exact flat-ground behavior.

Body frame convention:
  x = forward, y = left, yaw (psi) positive counter-clockwise.
Wheel order is always [FL, FR, RL, RR].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CarSpec, SimSpec

# wheel index constants
FL, FR, RL, RR = 0, 1, 2, 3
FRONT = np.array([FL, FR])
REAR = np.array([RL, RR])
STEERED = FRONT

G = 9.81


@dataclass
class Controls:
    """Normalised driver / agent commands."""
    steer: float = 0.0       # [-1, 1]  (+ = left)
    throttle: float = 0.0    # [0, 1]
    brake: float = 0.0       # [0, 1]
    clutch: float = 1.0      # [0, 1]  (1 = fully engaged)
    handbrake: float = 0.0   # [0, 1]
    shift_up: bool = False   # rising-edge request
    shift_down: bool = False

    def clamped(self) -> "Controls":
        return Controls(
            steer=float(np.clip(self.steer, -1.0, 1.0)),
            throttle=float(np.clip(self.throttle, 0.0, 1.0)),
            brake=float(np.clip(self.brake, 0.0, 1.0)),
            clutch=float(np.clip(self.clutch, 0.0, 1.0)),
            handbrake=float(np.clip(self.handbrake, 0.0, 1.0)),
            shift_up=bool(self.shift_up),
            shift_down=bool(self.shift_down),
        )


def _pacejka(s: np.ndarray, B: float, C: float, E: float) -> np.ndarray:
    """Signed, normalised Magic-Formula shape (odd in `s`); multiply by
    D = mu*Fz for force. `s` is slip ratio (long.) or tan(slip angle) (lat.)."""
    bs = B * s
    return np.sin(C * np.arctan(bs - E * (bs - np.arctan(bs))))


class Vehicle:
    """A single car's dynamic state + integrator."""

    def __init__(self, spec: CarSpec, sim: SimSpec | None = None):
        self.spec = spec
        self.sim = sim or SimSpec()
        self.max_steer_angle = np.radians(spec.steer_angle_max_deg)
        self.max_steer_rate = np.radians(spec.steer_rate_deg_s)
        # external surface grip multiplier (1.0 = tarmac); set per-step by the
        # env/app from the track so off-track is slippery.
        self.surface_grip = 1.0
        self.reset()

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
              speed: float = 0.0):
        s = self.spec
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.vx = self.vy = self.r = 0.0          # body velocities + yaw rate
        self.ax = self.ay = 0.0                    # remembered for weight transfer
        self.wheel_w = np.zeros(4)                 # wheel angular speeds (rad/s)
        self.Fz = self._static_loads()             # vertical loads (lag-filtered)
        self.engine_w = self._rpm_to_w(s.idle_rpm)
        self.gear = 1
        self.boost = 0.0
        self.limiter_cut = False
        self.center_split_front = float(np.clip(s.center_split_front, 0.0, 1.0))
        self.rear_steer_angle = 0.0
        # road-plane / vertical state (PHYSICS_3D_PLAN Stage 2) — all zero on
        # flat ground; callers that never set_road() keep pre-hills behavior
        self.grade_body = 0.0          # road slope under the nose (+ = climbing)
        self.bank_body = 0.0           # road slope under the left side (+ = left up)
        self.road_z = 0.0              # road height beneath the car
        self.road_vcurv = 0.0          # vertical curvature (- = crest)
        self.z = 0.0                   # car height (== road_z while grounded)
        self.vz = 0.0                  # vertical velocity
        self.pitch = 0.0               # body pitch (rad, kinematic output)
        self.roll = 0.0                # body roll (rad, kinematic output)
        self.airborne = False          # Stage 3 flips this; grounded until then
        self.air_time = 0.0
        self.landing_g = 0.0
        self._vert_raw = 1.0           # unclipped crest load factor (<=0 = takeoff)
        self._contact = 1.0            # 1.0 grounded / 0.0 airborne (friction gate)
        self._grav_fx = 0.0
        self._grav_fy = 0.0
        self._impact_Fz = np.zeros(4)  # decaying landing load spike (N per wheel)
        # raw road-plane values (track frame) — kept fresh even while airborne
        # so landing can re-project the surface under the touchdown point
        self._road_grade = 0.0
        self._road_bank = 0.0
        self._road_heading = 0.0
        # optional rolling start: spin the wheels + engine up to match `speed` and
        # pick a sensible gear, so an episode can begin AT pace with no jolt.
        if speed > 0.1:
            self.vx = float(speed)
            w_wheel = speed / s.wheel_radius
            self.wheel_w[:] = w_wheel
            best_g, best_err = 1, 1e18
            for g in range(1, len(s.gear_ratios) + 1):
                ratio = s.gear_ratios[g - 1] * s.final_drive
                rpm = w_wheel * ratio * 60.0 / (2.0 * np.pi)
                err = abs(rpm - 0.45 * s.redline_rpm) + (1e6 if rpm > s.redline_rpm else 0)
                if err < best_err:
                    best_err, best_g = err, g
            self.gear = best_g
            ratio = s.gear_ratios[best_g - 1] * s.final_drive
            self.engine_w = max(self._rpm_to_w(s.idle_rpm), w_wheel * ratio)
        self.steer_angle = 0.0
        # lagged slips (tyre relaxation state)
        self.alpha_lag = np.zeros(4)               # slip angle (rad), lateral
        self.kappa_lag = np.zeros(4)               # slip ratio, longitudinal
        # telemetry / rendering
        self.wheel_Fx = np.zeros(4)                # body frame
        self.wheel_Fy = np.zeros(4)
        self.wheel_slip = np.zeros(4)
        self.wheel_sr = np.zeros(4)
        self.wheel_grip = np.zeros(4)              # |F| / (mu*Fz) in [0,1+]
        
        # multi-agent damage states
        self.engine_damage = 0.0
        self.aero_damage = 0.0
        self.steer_bias = 0.0
        self.turbo_broken = False
        self.drivetrain_broken = False
        self.wheel_damage = np.zeros(4)
        self.misfire_phase = 0.0
        self.damage_events = []                    # queue of text popups
        return self

    def _static_loads(self) -> np.ndarray:
        s = self.spec
        wf = s.mass * G * s.front_weight / 2.0
        wr = s.mass * G * (1.0 - s.front_weight) / 2.0
        return np.array([wf, wf, wr, wr])

    def set_road(self, grade: float, bank: float, heading: float, z: float,
                 vcurv: float = 0.0):
        """Per-step road-plane input from the track (same caller contract as
        `surface_grip`: the driving loop reads `trk.frame()` and feeds it in
        before `step()`).

        Projects the track-frame slope onto the car's body axes via the
        heading difference — a car sliding sideways down a climb feels the
        pull on its SIDE, not its nose. Small-angle treatment of the slope
        vector rotation; exact enough for |grade| <= 0.2. Pitch/roll are
        kinematic OUTPUTS (telemetry/sensors/rendering), not states.

        While AIRBORNE (Stage 3) only the road height refreshes for landing
        detection — a free body feels no surface, so slopes zero out and the
        flight integrator owns z/vz/pitch. Never calling this method at all
        means flat ground: exact pre-hills behavior.
        """
        self.road_z = float(z)
        self.road_vcurv = float(vcurv)
        self._road_grade = float(grade)
        self._road_bank = float(bank)
        self._road_heading = float(heading)
        if self.airborne:
            return
        rel = self.yaw - heading
        c, s = np.cos(rel), np.sin(rel)
        self.grade_body = grade * c - bank * s
        self.bank_body = grade * s + bank * c
        self.z = float(z)
        self.vz = self.vx * self.grade_body
        self.pitch = float(np.arctan(self.grade_body))
        self.roll = float(np.arctan(self.bank_body))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _rpm_to_w(self, rpm: float) -> float:
        return rpm * 2.0 * np.pi / 60.0

    @property
    def rpm(self) -> float:
        return self.engine_w * 60.0 / (2.0 * np.pi)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def slip_angle(self) -> float:
        """Chassis sideslip: angle between heading and velocity vector (rad)."""
        if self.speed < 0.3:
            return 0.0
        return float(np.arctan2(self.vy, abs(self.vx)))

    def get_obb(self) -> np.ndarray:
        """Returns the 4 corner points of the Oriented Bounding Box [4, 2].
        Order: Front-Left, Front-Right, Rear-Right, Rear-Left."""
        s = self.spec
        # Bounding box must exactly match the visual model dimensions (carart.py)
        HL = (s.wheelbase + 1.7) / 2.0
        HW = (s.track_width + 0.36) / 2.0
        
        front_x = HL
        rear_x = -HL
        hw = HW
        
        # local corners (x = forward, y = left)
        local_corners = np.array([
            [front_x, hw],    # FL
            [front_x, -hw],   # FR
            [rear_x, -hw],    # RR
            [rear_x, hw]      # RL
        ])
        c, s_ang = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s_ang], [s_ang, c]])
        return np.array([self.x, self.y]) + local_corners @ R.T

    def apply_damage(self, local_x: float, local_y: float, ke: float):
        """Distributes absorbed kinetic energy into mechanical component failures."""
        s = self.spec
        # Scale damage by chassis durability (higher durability = less damage)
        damage = (ke / 50000.0) / max(s.durability, 0.1)
        if damage < 0.05:
            return  # superficial scratch, no mechanical effect
            
        # 1. Frontal Impact (Radiator / Engine / Aero / Steering)
        if local_x > s.a * 0.5:
            if s.name == "mazda787b":
                self.aero_damage = min(1.0, self.aero_damage + damage * 1.5)
                self.damage_events.append(f"{s.name}: Splitter Shattered!")
            if s.name == "lr4" or s.name == "f150":
                pass # massive steel front, shrugs it off
            else:
                self.engine_damage = min(1.0, self.engine_damage + damage * 0.5)
                if damage > 0.5:
                    self.damage_events.append(f"{s.name}: Radiator Crushed / Misfire!")
                else:
                    self.damage_events.append(f"{s.name}: Frontal Impact")
                    
        # 2. Side Impact (Tie-Rods / Suspension / Turbos)
        elif abs(local_y) > s.half_track * 0.8:
            # Front side impact bends tie rods
            if local_x > 0:
                self.steer_bias += (np.random.rand() - 0.5) * damage * 0.3
                self.damage_events.append(f"{s.name}: Tie-Rod Bent!")
            
            # Specific car vulnerabilities on side impact
            if s.name == "supra" and damage > 0.4 and not self.turbo_broken:
                self.turbo_broken = True
                self.damage_events.append(f"{s.name}: Intercooler Piping Severed (Boost Lost)!")
            elif s.name == "rx7" and damage > 0.3 and not self.turbo_broken:
                self.turbo_broken = True
                self.damage_events.append(f"{s.name}: Sequential Turbo Valve Stuck!")
                
            # Corner-specific suspension ruin
            if local_x > 0 and local_y > 0: self.wheel_damage[FL] = min(1.0, self.wheel_damage[FL] + damage)
            if local_x > 0 and local_y < 0: self.wheel_damage[FR] = min(1.0, self.wheel_damage[FR] + damage)
            if local_x < 0 and local_y > 0: self.wheel_damage[RL] = min(1.0, self.wheel_damage[RL] + damage)
            if local_x < 0 and local_y < 0: self.wheel_damage[RR] = min(1.0, self.wheel_damage[RR] + damage)

        # 3. Rear Impact (Drivetrain / Axles)
        if local_x < -s.b * 0.5:
            if s.name == "f150":
                # Solid rear axle bends, throwing alignment out
                self.steer_bias += (np.random.rand() - 0.5) * damage * 0.2
                self.damage_events.append(f"{s.name}: Solid Axle Bent (Crabbing)!")
            elif s.name == "skyline" and not self.drivetrain_broken and damage > 0.6:
                self.drivetrain_broken = True
                self.damage_events.append(f"{s.name}: ATTESA Transfer Case Shattered!")
            else:
                self.damage_events.append(f"{s.name}: Rear Impact")

    @property
    def total_ratio(self) -> float:
        return self.spec.gear_ratios[self.gear - 1] * self.spec.final_drive

    def shift_up(self):
        if self.gear < len(self.spec.gear_ratios):
            self.gear += 1

    def shift_down(self):
        if self.gear > 1:
            self.gear -= 1

    def _engine_torque(self, throttle: float) -> float:
        s = self.spec
        rpm = self.rpm
        # Hysteresis prevents the hard cut from chattering sample-to-sample.
        # RaceBox normally shifts before this; it is the final mechanical guard.
        if self.limiter_cut and rpm <= s.cutoff_rpm * 0.975:
            self.limiter_cut = False
        elif rpm >= s.cutoff_rpm:
            self.limiter_cut = True
        if self.limiter_cut:
            return 0.0
            
        # Severe engine damage raises stall RPM and randomly kills torque (misfire)
        if self.engine_damage > 0.5:
            if rpm < s.idle_rpm * 1.5:
                return 0.0 # stalls out easily
            self.misfire_phase += np.random.rand() * 0.5
            if np.sin(self.misfire_phase) > 0.8:
                return 0.0 # cylinder misfire
                
        rpms = np.array([p[0] for p in s.torque_curve])
        tqs = np.array([p[1] for p in s.torque_curve])
        base = float(np.interp(rpm, rpms, tqs))
        
        # apply global torque loss
        base *= (1.0 - self.engine_damage * 0.7) 
        
        boost_mult = s.boost_floor + (1.0 - s.boost_floor) * self.boost
        return base * throttle * boost_mult

    def _update_boost(self, throttle: float, dt: float):
        s = self.spec
        if s.boost_floor >= 1.0 or self.turbo_broken: # naturally aspirated OR broken turbo
            self.boost = 0.0
            return
        rpm_factor = np.clip((self.rpm - 1800.0) / 3000.0, 0.0, 1.0)
        target = throttle * rpm_factor
        tau = s.spool_up_tau if target > self.boost else s.spool_down_tau
        self.boost += (dt / (tau + dt)) * (target - self.boost)
        self.boost = float(np.clip(self.boost, 0.0, 1.0))

    def _rear_steer_angle(self) -> float:
        """Small Super-HICAS-style rear steer. Most cars leave this disabled."""
        s = self.spec
        max_angle = np.radians(s.rear_steer_max_deg)
        if max_angle <= 0.0:
            return 0.0
        speed_blend = np.clip(self.speed / max(s.rear_steer_transition_mps, 1e-3), 0.0, 1.0)
        gain = (
            (1.0 - speed_blend) * s.rear_steer_low_speed_gain
            + speed_blend * s.rear_steer_high_speed_gain
        )
        return float(np.clip(self.steer_angle * gain, -max_angle, max_angle))

    def _front_torque_split(self, controls: Controls, sub_dt: float) -> float:
        """Current front-axle torque fraction.

        Static AWD cars use CarSpec.center_split_front directly. ATTESA-style
        cars ramp from rear-drive toward front assist when launch demand, rear
        slip, or g-sensor demand rises.
        """
        s = self.spec
        if s.drive_layout != "awd" or self.drivetrain_broken:
            self.center_split_front = 0.0 if s.drive_layout == "rwd" or self.drivetrain_broken else 1.0
            return self.center_split_front
        if s.center_diff != "attesa":
            self.center_split_front = float(np.clip(s.center_split_front, 0.0, 1.0))
            return self.center_split_front

        front_kappa = max(0.0, float(np.mean(self.kappa_lag[FRONT])))
        rear_kappa = max(0.0, float(np.mean(self.kappa_lag[REAR])))
        rear_slip_need = max(0.0, rear_kappa - front_kappa)
        launch_need = controls.throttle * max(0.0, 1.0 - self.speed / 10.0)
        g_need = 0.65 * max(0.0, self.ax / G) + 0.35 * abs(self.ay / G)

        target = (
            s.attesa_front_min
            + s.attesa_launch_split * launch_need
            + s.attesa_slip_gain * rear_slip_need
            + s.attesa_g_gain * g_need
        )
        if controls.brake > 0.05 or controls.handbrake > 0.05:
            target *= 0.35
        target = float(np.clip(target, s.attesa_front_min, s.attesa_front_max))
        a = sub_dt / (s.attesa_response_tau + sub_dt)
        self.center_split_front += a * (target - self.center_split_front)
        return float(np.clip(self.center_split_front, s.attesa_front_min, s.attesa_front_max))

    # ------------------------------------------------------------------ #
    # combined-slip tyre model
    # ------------------------------------------------------------------ #
    def _tyre_forces(self, kappa: np.ndarray, tan_alpha: np.ndarray,
                     D: np.ndarray):
        """Pure-slip Pacejka forces coupled through a friction ellipse.

        Longitudinal and lateral forces are computed on their own curves (which
        have different stiffnesses), then jointly limited so the resultant can't
        exceed the friction budget D = mu*Fz. Coupling therefore bites near the
        limit: flooring the throttle drives Fx up, the ellipse scales both down,
        and the rear loses lateral grip -> controllable power-on oversteer.
        Returns (Fx, Fy) in wheel frame.
        """
        s = self.spec
        Fx0 = D * _pacejka(kappa, s.pacejka_long.B, s.pacejka_long.C, s.pacejka_long.E)
        Fy0 = -D * _pacejka(tan_alpha, s.pacejka_lat.B, s.pacejka_lat.C, s.pacejka_lat.E)
        total = np.hypot(Fx0, Fy0)
        scale = np.where(total > D, D / np.maximum(total, 1e-6), 1.0)
        return Fx0 * scale, Fy0 * scale

    # ------------------------------------------------------------------ #
    # main step
    # ------------------------------------------------------------------ #
    def step(self, controls: Controls, dt: float | None = None):
        s = self.spec
        sim = self.sim
        dt = sim.dt if dt is None else dt
        c = controls.clamped()

        if c.shift_up:
            self.shift_up()
        elif c.shift_down:
            self.shift_down()

        # rate-limited steering toward the commanded angle (with permanent damage bias)
        target_steer = (c.steer * self.max_steer_angle) + self.steer_bias
        
        # severe tie-rod damage binds the steering rack (slower turning)
        d_max = (self.max_steer_rate * (1.0 - np.clip(self.steer_bias, 0.0, 0.8))) * dt
        self.steer_angle += float(np.clip(target_steer - self.steer_angle, -d_max, d_max))

        # contact-patch geometry + velocities (body frame)
        xpos = np.array([s.a, s.a, -s.b, -s.b])
        ypos = np.array([s.half_track, -s.half_track, s.half_track, -s.half_track])
        vpx = self.vx - self.r * ypos
        vpy = self.vy + self.r * xpos

        # rotate front patches into their steered wheel frame
        delta = np.zeros(4)
        delta[STEERED] = self.steer_angle
        self.rear_steer_angle = self._rear_steer_angle()
        delta[REAR] = self.rear_steer_angle
        cd, sd = np.cos(delta), np.sin(delta)
        v_long = vpx * cd + vpy * sd
        v_lat = -vpx * sd + vpy * cd
        denom = np.maximum(np.abs(v_long), sim.velocity_eps)

        # --- slip-angle relaxation (tyre force lag) ---
        # airborne (_contact = 0): no contact patch, so the slip state relaxes
        # toward zero instead of the kinematic angle. x*1.0 is exact on ground.
        alpha_inst = np.arctan2(v_lat, denom) * self._contact
        tau_lat = s.relaxation_length / np.maximum(np.abs(v_long), sim.relax_speed_floor)
        a_lat = dt / (tau_lat + dt)
        self.alpha_lag += a_lat * (alpha_inst - self.alpha_lag)
        self.wheel_slip = self.alpha_lag.copy()

        # --- vertical loads (transfer + downforce + lag) ---
        self._update_loads(dt)
        mu = self._mu_effective() * self.surface_grip
        
        # Apply corner-specific tire/suspension damage (drags down the mu limit)
        mu *= (1.0 - self.wheel_damage * 0.8)
        
        # _contact zeroes the friction budget the instant the car leaves the
        # ground (the Fz lag alone would keep phantom grip for ~suspension_tau)
        D = mu * self.Fz * self._contact

        # --- drivetrain + combined tyre forces (sub-stepped) ---
        Fx, Fy = self._drivetrain(c, v_long, denom, D, dt)
        self.wheel_grip = np.hypot(Fx, Fy) / np.maximum(D, 1.0)

        # rotate tyre forces back to body frame
        Fbx = Fx * cd - Fy * sd
        Fby = Fx * sd + Fy * cd
        self.wheel_Fx, self.wheel_Fy = Fbx, Fby

        # --- aero + rolling resistance (oppose full velocity vector) ---
        drag_x = 0.5 * s.air_density * s.drag_area * self.vx * abs(self.vx)
        drag_y = 0.5 * s.air_density * s.side_drag_area * self.vy * abs(self.vy)
        spd = self.speed
        rr_mag = (s.rolling_resistance * float(self.Fz.sum())
                  * min(1.0, spd / 0.5) * self._contact)
        rr_x = rr_mag * (self.vx / spd) if spd > 1e-3 else 0.0
        rr_y = rr_mag * (self.vy / spd) if spd > 1e-3 else 0.0

        # --- aggregate forces & moments ---
        # 2.5D slope gravity: appended terms that are exactly -0.0 on flat
        # ground (byte-identity). ax/ay therefore INCLUDE the gravity pull —
        # what an accelerometer should feel, and what ATTESA g_need sees.
        self._grav_fx = -s.mass * G * self.grade_body
        self._grav_fy = -s.mass * G * self.bank_body
        Fx_tot = float(Fbx.sum()) - drag_x - rr_x + self._grav_fx
        Fy_tot = float(Fby.sum()) - drag_y - rr_y + self._grav_fy
        Mz = float((xpos * Fby - ypos * Fbx).sum())

        self.ax = Fx_tot / s.mass
        self.ay = Fy_tot / s.mass
        
        # Stable exact rotation for Coriolis forces (prevents Euler explosion at high yaw rates)
        angle = self.r * dt
        c_r, s_r = np.cos(angle), np.sin(angle)
        vx_rot = self.vx * c_r + self.vy * s_r
        vy_rot = -self.vx * s_r + self.vy * c_r
        
        self.vx = float(vx_rot + self.ax * dt)
        self.vy = float(vy_rot + self.ay * dt)
        self.r += float((Mz / s.yaw_inertia) * dt)

        if self.speed < 0.05 and c.throttle < 0.01 and not self.airborne:
            self.vx *= 0.4
            self.vy *= 0.4
            self.r *= 0.4

        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (self.vx * cy - self.vy * sy) * dt
        self.y += (self.vx * sy + self.vy * cy) * dt
        self.yaw += self.r * dt

        # --- vertical mode transitions + flight integration (Stage 3) ---
        self._update_vertical(dt)
        return self

    # ------------------------------------------------------------------ #
    # airborne: takeoff / ballistic flight / landing (PHYSICS_3D_PLAN Stage 3)
    # ------------------------------------------------------------------ #
    def _update_vertical(self, dt: float):
        """Mode transitions for the vertical degree of freedom.

        Takeoff is EMERGENT: when the v^2 crest term drives the unclipped
        load factor to <= 0 (`_vert_raw`, set in _update_loads), the road can
        no longer hold the car down — unsatisfiable on flat ground, where the
        factor is identically 1.0. Flight is ballistic: planar momentum is
        conserved (only aero drag acts — tyres/rr are gated off by _contact),
        vz integrates under -g, yaw rate persists (Mz = 0 falls out of zero
        tyre forces). Landing snaps back to the road and absorbs the relative
        vertical speed through a decaying suspension load spike over
        landing_tau; the grip consequences (load-sensitive mu, slip-ratio
        spike) are emergent, not scripted.
        """
        s = self.spec
        if not self.airborne:
            if self._vert_raw <= 0.0:
                # the crest threw us: ramp angle sets the launch vz
                self.airborne = True
                self._contact = 0.0
                self.air_time = 0.0
                self.landing_g = 0.0
                self.vz = self.vx * self.grade_body
                self.grade_body = 0.0      # a free body feels no surface
                self.bank_body = 0.0
            return

        # --- flight ---
        self.vz -= G * dt
        self.z += self.vz * dt
        self.air_time += dt
        # nose follows the flight arc; roll eases level; wheels keep spinning
        # (driven ones obey the drivetrain — free-rev off the jump)
        flight = float(np.arctan2(self.vz, max(self.speed, 1.0)))
        rate = s.air_pitch_rate * dt
        self.pitch += float(np.clip(flight - self.pitch, -rate, rate))
        self.roll += float(np.clip(-self.roll, -rate, rate))
        self.wheel_w *= 1.0 / (1.0 + 0.05 * dt)   # faint bearing drag

        # --- landing? (descending onto the road surface under the car) ---
        rel = self.yaw - self._road_heading
        c0, s0 = np.cos(rel), np.sin(rel)
        gb = self._road_grade * c0 - self._road_bank * s0
        bb = self._road_grade * s0 + self._road_bank * c0
        vz_road = self.vx * gb                     # surface falls/rises ahead
        if self.z <= self.road_z and self.vz <= vz_road + 1e-9:
            dv = max(0.0, vz_road - self.vz)       # impact severity (m/s)
            pitch_air = self.pitch
            self.airborne = False
            self._contact = 1.0
            self.z = self.road_z
            self.grade_body, self.bank_body = gb, bb
            self.vz = vz_road
            self.pitch = float(np.arctan(gb))
            self.roll = float(np.arctan(bb))
            # suspension impact: decaying load spike, clamped, biased toward
            # whichever axle the car landed on (nose-low hammers the fronts)
            tau = max(s.landing_tau, 1e-3)
            total = min(s.mass * dv / tau, s.max_landing_load * s.mass * G)
            front = float(np.clip(0.5 - 1.5 * (pitch_air - self.pitch),
                                  0.2, 0.8))
            self._impact_Fz = total * np.array(
                [front / 2.0, front / 2.0,
                 (1.0 - front) / 2.0, (1.0 - front) / 2.0])
            self.landing_g = dv / (G * tau)

    # ------------------------------------------------------------------ #
    # loads / friction
    # ------------------------------------------------------------------ #
    def _update_loads(self, dt: float):
        s = self.spec
        long_transfer = s.mass * self.ax * s.cg_height / s.wheelbase   # front->rear
        lat_transfer = s.mass * self.ay * s.cg_height / s.track_width  # left->right
        # roll-stiffness distribution of lateral transfer between the axles
        ltf_front = lat_transfer * s.roll_front_frac
        ltf_rear = lat_transfer * (1.0 - s.roll_front_frac)
        # speed-dependent downforce (uses forward speed, heavily reduced by aero damage)
        df = 0.5 * s.air_density * s.downforce_ClA * self.vx * self.vx * (1.0 - self.aero_damage)
        
        # front splitter damage shifts aero balance severely rearward
        actual_aero_balance = s.aero_balance * (1.0 - self.aero_damage)
        df_front = df * actual_aero_balance
        df_rear = df * (1.0 - actual_aero_balance)

        target = self._static_loads()
        target[FRONT] -= long_transfer / 2.0
        target[REAR] += long_transfer / 2.0
        target[FL] -= ltf_front
        target[FR] += ltf_front
        target[RL] -= ltf_rear
        target[RR] += ltf_rear
        target[FRONT] += df_front / 2.0
        target[REAR] += df_rear / 2.0
        target = np.maximum(target, 0.0)

        # --- 2.5D slope/crest terms (PHYSICS_3D_PLAN Stage 2) ---
        # appended AFTER the existing math: every term is an exact identity
        # when grade=bank=vcurv=0, so flat ground stays byte-identical.
        gb, bb = self.grade_body, self.bank_body
        cosN = 1.0 / np.sqrt(1.0 + gb * gb + bb * bb)   # normal-load factor
        # v^2 crest/dip term (vcurv NEGATIVE at crests): light over crests,
        # heavy in dips. Floor 0.0 — a fast-enough crest FULLY unloads the
        # car; Stage 3 reads _vert_raw <= 0 as the takeoff trigger.
        vert_raw = 1.0 + self.road_vcurv * self.vx * self.vx / G
        self._vert_raw = vert_raw
        target *= cosN * np.clip(vert_raw, 0.0, 1.7)
        # static slope transfer — explicit, NEVER via the ax-based term above
        # (gravity already lives in ax; routing it through dynamic transfer
        # would shift load the wrong way). Climb loads the rear; left-edge-up
        # loads the right side, split by roll stiffness like lateral transfer.
        gt = s.mass * G * gb * s.cg_height / s.wheelbase
        target[FRONT] -= gt / 2.0
        target[REAR] += gt / 2.0
        bt = s.mass * G * bb * s.cg_height / s.track_width
        btf = bt * s.roll_front_frac
        btr = bt * (1.0 - s.roll_front_frac)
        target[FL] -= btf
        target[FR] += btf
        target[RL] -= btr
        target[RR] += btr
        target = np.maximum(target, 0.0)

        # airborne (Stage 3): loads target zero (suspension extends over its
        # lag); on touchdown the decaying impact spike rides on top, feeding
        # the load-sensitive mu — hard landings cost grip physically.
        target *= self._contact
        target += self._impact_Fz
        self._impact_Fz *= s.landing_tau / (s.landing_tau + dt)

        a = dt / (s.suspension_tau + dt)
        self.Fz += a * (target - self.Fz)
        self.Fz = np.maximum(self.Fz, 0.0)

    def _mu_effective(self) -> np.ndarray:
        s = self.spec
        nominal = s.mass * G / 4.0
        mu = s.mu * (1.0 - s.load_sensitivity * (self.Fz - nominal))
        return np.maximum(mu, 0.1)

    # ------------------------------------------------------------------ #
    # drivetrain (sub-stepped) + longitudinal slip relaxation
    # ------------------------------------------------------------------ #
    def _drivetrain(self, c: Controls, v_long, denom, D, dt):
        """Evolve engine + wheel spin and resolve combined tyre forces.

        Slip angle (relaxed in step()) is held; slip ratio is relaxed and the
        combined tyre force is recomputed each substep, so wheelspin/lock-up
        and their effect on lateral grip resolve at the driveline's timescale.
        """
        s = self.spec
        n = self.sim.drivetrain_substeps
        sub = dt / n
        r = s.wheel_radius
        tan_alpha = np.tan(self.alpha_lag)

        brake_axle = np.array([
            c.brake * s.max_brake_torque * s.brake_bias / 2.0,
            c.brake * s.max_brake_torque * s.brake_bias / 2.0,
            c.brake * s.max_brake_torque * (1.0 - s.brake_bias) / 2.0,
            c.brake * s.max_brake_torque * (1.0 - s.brake_bias) / 2.0,
        ])
        brake_axle[REAR] += c.handbrake * s.handbrake_torque / 2.0

        self._update_boost(c.throttle, dt)

        # Short longitudinal slip relaxation. dFx/d(omega) ~ 1/v, so at low speed
        # the slip ratio is violently sensitive to wheel-spin jitter; lagging it
        # damps that singularity (which otherwise lets the stiff clutch coupling
        # pump energy and creep/accelerate the car). Kept short so launches stay
        # crisp.
        tau_long = s.relaxation_length_long / np.maximum(np.abs(v_long), self.sim.relax_speed_floor)
        a_long = sub / (tau_long + sub)
        roll_w = v_long / r

        # ---- FAST PATH: RWD + clutch (the existing cars) ----
        # Kept verbatim so supra/rx7 stay bit-identical (and their trained
        # checkpoints valid). The general path below handles FWD/AWD + torque
        # converter. Undriven wheels with no brake free-roll: pinning them to
        # ground speed avoids spurious low-speed slip-ratio forces.
        if s.drive_layout == "rwd" and s.coupling == "clutch":
            free_roll = np.array([True, True, False, False]) & (brake_axle <= 1e-6)
            if self.airborne:              # no ground: nothing to pin wheels to
                free_roll[:] = False
            Fx = np.zeros(4)
            Fy = np.zeros(4)
            for _ in range(n):
                self.wheel_w[free_roll] = roll_w[free_roll]
                kappa_inst = (np.clip((self.wheel_w * r - v_long) / denom,
                                      -4.0, 4.0) * self._contact)
                self.kappa_lag += a_long * (kappa_inst - self.kappa_lag)
                Fx, Fy = self._tyre_forces(self.kappa_lag, tan_alpha, D)
                self.wheel_sr = self.kappa_lag.copy()

                # engine + clutch
                Te = self._engine_torque(c.throttle) - s.engine_friction * self.engine_w
                idle_w = self._rpm_to_w(s.idle_rpm)
                if self.engine_w < idle_w:
                    Te += min(s.idle_torque_max, s.idle_gain * (idle_w - self.engine_w))
                drive_w = 0.5 * (self.wheel_w[RL] + self.wheel_w[RR])
                d_w = self.engine_w - drive_w * self.total_ratio
                Tc = c.clutch * s.clutch_capacity * np.tanh(d_w / s.clutch_slip_ref)
                self.engine_w += sub * (Te - Tc) / s.engine_inertia
                self.engine_w = max(self.engine_w, 0.0)   # can bog to a stall, never reverse

                # driveline torque to rear wheels through gearbox, split by LSD
                T_drive = Tc * self.total_ratio * s.drivetrain_efficiency
                t_lsd = s.lsd_coef * (self.wheel_w[RL] - self.wheel_w[RR])
                T_wheel = np.zeros(4)
                T_wheel[RL] = T_drive / 2.0 - t_lsd
                T_wheel[RR] = T_drive / 2.0 + t_lsd

                # wheel spin: drive - tyre reaction; then brakes clamp through zero
                self.wheel_w += sub * (T_wheel - Fx * r) / s.wheel_inertia
                dw_brake = sub * brake_axle / s.wheel_inertia
                self.wheel_w = np.where(
                    np.abs(self.wheel_w) <= dw_brake,
                    0.0,
                    self.wheel_w - np.sign(self.wheel_w) * dw_brake,
                )
            return Fx, Fy

        # ---- GENERAL PATH: FWD / AWD and/or torque-converter coupling ----
        driven = self._driven_mask()
        free_roll = (~driven) & (brake_axle <= 1e-6)
        if self.airborne:                  # no ground: nothing to pin wheels to
            free_roll[:] = False
        Fx = np.zeros(4)
        Fy = np.zeros(4)
        for _ in range(n):
            self.wheel_w[free_roll] = roll_w[free_roll]
            kappa_inst = (np.clip((self.wheel_w * r - v_long) / denom,
                                  -4.0, 4.0) * self._contact)
            self.kappa_lag += a_long * (kappa_inst - self.kappa_lag)
            Fx, Fy = self._tyre_forces(self.kappa_lag, tan_alpha, D)
            self.wheel_sr = self.kappa_lag.copy()

            # engine torque (+ idle governor)
            Te = self._engine_torque(c.throttle) - s.engine_friction * self.engine_w
            idle_w = self._rpm_to_w(s.idle_rpm)
            if self.engine_w < idle_w:
                Te += min(s.idle_torque_max, s.idle_gain * (idle_w - self.engine_w))

            # input-shaft speed seen through the gearbox. ATTESA is rear-drive
            # primary, with a transfer clutch feeding the front axle as needed.
            if s.drive_layout == "awd" and s.center_diff == "attesa":
                drive_w = 0.5 * (self.wheel_w[RL] + self.wheel_w[RR])
            else:
                drive_w = float(np.mean(self.wheel_w[driven]))
            omega_tur = drive_w * self.total_ratio

            # --- coupling: clutch OR torque converter -> T_in (gearbox input) ---
            if s.coupling == "torque_converter":
                w_imp = max(self.engine_w, 1.0)
                sr = float(np.clip(omega_tur / w_imp, 0.0, 1.0))
                T_pump = s.tc_capacity * w_imp * w_imp * max(0.0, 1.0 - sr * sr)
                TR = 1.0 + (s.tc_mult_max - 1.0) * max(0.0, 1.0 - sr / max(s.tc_coupling_sr, 1e-3))
                T_in = T_pump * TR
                if sr > s.tc_lockup_sr:                       # lockup clutch blend
                    lock = (sr - s.tc_lockup_sr) / max(1.0 - s.tc_lockup_sr, 1e-3)
                    lock = float(np.clip(lock, 0.0, 1.0)) * c.clutch
                    Tc_lock = s.clutch_capacity * np.tanh((self.engine_w - omega_tur) / s.clutch_slip_ref)
                    T_pump = (1.0 - lock) * T_pump + lock * Tc_lock
                    T_in = (1.0 - lock) * T_in + lock * Tc_lock
                self.engine_w += sub * (Te - T_pump) / s.engine_inertia
            else:                                             # clutch
                d_w = self.engine_w - omega_tur
                Tc = c.clutch * s.clutch_capacity * np.tanh(d_w / s.clutch_slip_ref)
                T_in = Tc
                self.engine_w += sub * (Te - Tc) / s.engine_inertia
            self.engine_w = max(self.engine_w, 0.0)

            T_drive = T_in * self.total_ratio * s.drivetrain_efficiency

            # --- center diff / transfer case ---
            front_split = self._front_torque_split(c, sub)
            front_axle_T = T_drive * front_split
            rear_axle_T = T_drive * (1.0 - front_split)
            if s.drive_layout == "awd" and s.center_diff in ("lsd", "locked"):
                w_front = 0.5 * (self.wheel_w[FL] + self.wheel_w[FR])
                w_rear = 0.5 * (self.wheel_w[RL] + self.wheel_w[RR])
                coef = s.center_lsd_coef * (8.0 if s.center_diff == "locked" else 1.0)
                t_center = coef * (w_front - w_rear)   # front faster -> shove torque rearward
                front_axle_T -= t_center
                rear_axle_T += t_center

            # --- per-axle diffs ---
            T_wheel = np.zeros(4)
            if front_split > 1e-5:                         # front axle driven
                t_lsd_f = s.front_lsd_coef * (self.wheel_w[FL] - self.wheel_w[FR])
                T_wheel[FL] = front_axle_T / 2.0 - t_lsd_f
                T_wheel[FR] = front_axle_T / 2.0 + t_lsd_f
            if front_split < 1.0 - 1e-5:                   # rear axle driven
                t_lsd_r = s.lsd_coef * (self.wheel_w[RL] - self.wheel_w[RR])
                T_wheel[RL] = rear_axle_T / 2.0 - t_lsd_r
                T_wheel[RR] = rear_axle_T / 2.0 + t_lsd_r

            # wheel spin: drive - tyre reaction; then brakes clamp through zero
            self.wheel_w += sub * (T_wheel - Fx * r) / s.wheel_inertia
            dw_brake = sub * brake_axle / s.wheel_inertia
            self.wheel_w = np.where(
                np.abs(self.wheel_w) <= dw_brake,
                0.0,
                self.wheel_w - np.sign(self.wheel_w) * dw_brake,
            )
        return Fx, Fy

    def _driven_mask(self) -> np.ndarray:
        """Bool[4] (FL,FR,RL,RR) of which wheels receive drive torque, per layout."""
        layout = self.spec.drive_layout
        if layout == "awd":
            return np.array([True, True, True, True])
        if layout == "fwd":
            return np.array([True, True, False, False])
        return np.array([False, False, True, True])   # rwd

    # ------------------------------------------------------------------ #
    # telemetry snapshot
    # ------------------------------------------------------------------ #
    def telemetry(self) -> dict:
        return {
            "speed": self.speed,
            "speed_kmh": self.speed * 3.6,
            "rpm": self.rpm,
            "gear": self.gear,
            "boost": self.boost,
            "steer_deg": np.degrees(self.steer_angle),
            "slip_angle_deg": np.degrees(self.slip_angle),
            "Fz": self.Fz.copy(),
            "wheel_w": self.wheel_w.copy(),
            "wheel_slip_deg": np.degrees(self.wheel_slip),
            "wheel_sr": self.wheel_sr.copy(),
            "wheel_grip": self.wheel_grip.copy(),
            "center_split_front": self.center_split_front,
            "rear_steer_deg": np.degrees(self.rear_steer_angle),
            "surface_grip": self.surface_grip,
            "ax": self.ax,
            "ay": self.ay,
            "pos": (self.x, self.y),
            "yaw": self.yaw,
            "grade_body": self.grade_body,
            "bank_body": self.bank_body,
            "pitch": self.pitch,
            "roll": self.roll,
            "z": self.z,
            "vz": self.vz,
            "airborne": self.airborne,
            "landing_g": self.landing_g,
        }

# ------------------------------------------------------------------ #
# Multi-Agent Collision & Impulse Resolution
# ------------------------------------------------------------------ #
def _get_axes(obb: np.ndarray) -> np.ndarray:
    """Returns the two normalized edge normal axes of the OBB."""
    axes = np.zeros((2, 2))
    edge1 = obb[1] - obb[0]
    axes[0] = np.array([-edge1[1], edge1[0]])
    axes[0] /= np.linalg.norm(axes[0]) + 1e-8
    edge2 = obb[3] - obb[0]
    axes[1] = np.array([-edge2[1], edge2[0]])
    axes[1] /= np.linalg.norm(axes[1]) + 1e-8
    return axes

def _project_obb(obb: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    dots = obb @ axis
    return float(np.min(dots)), float(np.max(dots))

def sat_collision(obb1: np.ndarray, obb2: np.ndarray) -> tuple[bool, np.ndarray, float]:
    """Separating Axis Theorem. Returns (colliding, normal_vector, penetration_depth).
    If colliding, the normal points from obb2 to obb1."""
    axes = np.vstack([_get_axes(obb1), _get_axes(obb2)])
    min_overlap = float('inf')
    mtv = np.zeros(2)
    
    for axis in axes:
        min1, max1 = _project_obb(obb1, axis)
        min2, max2 = _project_obb(obb2, axis)
        
        if min1 > max2 or min2 > max1:
            return False, np.zeros(2), 0.0
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            mtv = axis
            
    center1 = np.mean(obb1, axis=0)
    center2 = np.mean(obb2, axis=0)
    if np.dot(mtv, center1 - center2) < 0:
        mtv = -mtv
        
    return True, mtv, min_overlap

def resolve_collisions(vehicles: list[Vehicle], dt: float):
    """Resolves OBB collisions between all active vehicles applying elastic impulses."""
    n = len(vehicles)
    if n < 2:
        return
        
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = vehicles[i], vehicles[j]
            if v1.airborne or v2.airborne:
                continue
                
            obb1 = v1.get_obb()
            obb2 = v2.get_obb()
            colliding, normal, depth = sat_collision(obb1, obb2)
            
            if not colliding:
                continue
                
            # Geometric Penetration Resolution (Push apart inversely by mass)
            total_mass = v1.spec.mass + v2.spec.mass
            ratio1 = v2.spec.mass / total_mass
            ratio2 = v1.spec.mass / total_mass
            
            push = normal * depth
            v1.x += push[0] * ratio1
            v1.y += push[1] * ratio1
            v2.x -= push[0] * ratio2
            v2.y -= push[1] * ratio2
            
            # Contact point approx: average of centers
            contact_pt = (np.mean(obb1, axis=0) + np.mean(obb2, axis=0)) / 2.0
            r1 = contact_pt - np.array([v1.x, v1.y])
            r2 = contact_pt - np.array([v2.x, v2.y])
            
            vel1_world = np.array([
                v1.vx * np.cos(v1.yaw) - v1.vy * np.sin(v1.yaw) - v1.r * r1[1],
                v1.vx * np.sin(v1.yaw) + v1.vy * np.cos(v1.yaw) + v1.r * r1[0]
            ])
            vel2_world = np.array([
                v2.vx * np.cos(v2.yaw) - v2.vy * np.sin(v2.yaw) - v2.r * r2[1],
                v2.vx * np.sin(v2.yaw) + v2.vy * np.cos(v2.yaw) + v2.r * r2[0]
            ])
            
            v_rel = vel1_world - vel2_world
            vel_along_normal = np.dot(v_rel, normal)
            
            if vel_along_normal > 0:
                continue
                
            restitution = 0.25  # car bodies absorb energy
            r1_cross_n = r1[0]*normal[1] - r1[1]*normal[0]
            r2_cross_n = r2[0]*normal[1] - r2[1]*normal[0]
            
            inv_mass1 = 1.0 / v1.spec.mass
            inv_mass2 = 1.0 / v2.spec.mass
            inv_I1 = 1.0 / v1.spec.yaw_inertia
            inv_I2 = 1.0 / v2.spec.yaw_inertia
            
            impulse_denom = (inv_mass1 + inv_mass2 + 
                             (r1_cross_n**2) * inv_I1 + 
                             (r2_cross_n**2) * inv_I2)
                             
            j_imp = -(1.0 + restitution) * vel_along_normal / impulse_denom
            impulse_vec = j_imp * normal
            
            # Apply impulse
            dv1_world = impulse_vec * inv_mass1
            dv2_world = -impulse_vec * inv_mass2
            
            c1, s1 = np.cos(v1.yaw), np.sin(v1.yaw)
            v1.vx += dv1_world[0]*c1 + dv1_world[1]*s1
            v1.vy += -dv1_world[0]*s1 + dv1_world[1]*c1
            v1.r += r1_cross_n * j_imp * inv_I1
            
            c2, s2 = np.cos(v2.yaw), np.sin(v2.yaw)
            v2.vx += dv2_world[0]*c2 + dv2_world[1]*s2
            v2.vy += -dv2_world[0]*s2 + dv2_world[1]*c2
            v2.r -= r2_cross_n * j_imp * inv_I2
            
            # --- Damage Calculation ---
            # KE = 0.5 * m * dv^2 (using the relative closing velocity along normal)
            # Actually, total KE absorbed is related to the impulse magnitude
            ke_absorbed = 0.5 * abs(j_imp) * abs(vel_along_normal)
            
            # Convert world contact point to local body coords to see where they got hit
            r1_local_x = r1[0]*c1 + r1[1]*s1
            r1_local_y = -r1[0]*s1 + r1[1]*c1
            r2_local_x = r2[0]*c2 + r2[1]*s2
            r2_local_y = -r2[0]*s2 + r2[1]*c2
            
            v1.apply_damage(r1_local_x, r1_local_y, ke_absorbed)
            v2.apply_damage(r2_local_x, r2_local_y, ke_absorbed)
