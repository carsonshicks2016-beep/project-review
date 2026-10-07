"""
Four-wheel vehicle dynamics for Supra-AI.

A proper sim model, not a lumped "bicycle":

  * FOUR independent contact patches (FL, FR, RL, RR), each with its own
    vertical load, slip angle, slip ratio and grip;
  * LATERAL *and* LONGITUDINAL weight transfer with a suspension time-lag, so
    the weight *settles* instead of snapping -- that lag is the feel of mass;
  * LOAD-SENSITIVE tyres (peak grip rises slower than load) so transferring
    weight costs total grip -> smoothness is rewarded;
  * SLIP-RATIO longitudinal tyres with real wheel rotational inertia -> genuine
    wheelspin and brake lockup;
  * an engine FLYWHEEL with its own inertia and a brain-controllable CLUTCH
    (clutch capacity model) -> launches, rev-matching, and CLUTCH-KICK drifts;
  * brain-controllable SEQUENTIAL GEARBOX (or auto) + limited-slip diff;
  * 2JZ-GTE torque curve + sequential twin-turbo;
  * SELF-ALIGNING TORQUE exposed as steering feel;
  * human-style RATE-LIMITED steering / throttle / brake.

The stiff engine/clutch/wheel rotational dynamics are integrated with extra
sub-steps (sim.drivetrain_substeps) for stability.

Body frame: +x forward, +y LEFT, +yaw turns left.  Wheels: 0=FL 1=FR 2=RL 3=RR.
"""

from __future__ import annotations
import math
import numpy as np

from .config import CarSpec, SimSpec

_TWO_PI = 2.0 * math.pi


def _pac(s, B, C, E):
    """Pacejka magic formula, normalised to [-1, 1]."""
    bs = B * s
    return math.sin(C * math.atan(bs - E * (bs - math.atan(bs))))


class Vehicle:
    def __init__(self, car: CarSpec, sim: SimSpec):
        self.car = car
        self.sim = sim
        self._rpm_pts = np.array([p[0] for p in car.torque_curve], dtype=np.float64)
        self._trq_pts = np.array([p[1] for p in car.torque_curve], dtype=np.float64)

        lf, lr, t = car.lf, car.lr, car.track_width
        self.wa = np.array([lf, lf, -lr, -lr])
        self.wb = np.array([t / 2, -t / 2, t / 2, -t / 2])
        self.is_front = np.array([True, True, False, False])
        self._idle_omega = car.idle_rpm * _TWO_PI / 60.0
        self._rev_omega = car.rev_limit_rpm * _TWO_PI / 60.0
        self.reset(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ #
    def reset(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self.vx = self.vy = self.r = 0.0
        self.ax = self.ay = 0.0
        # powertrain
        self.engine_omega = self._idle_omega
        self.rpm = self.car.idle_rpm
        self.gear = 0
        self.boost = 0.0
        self.clutch = 1.0
        self.shift_timer = 0.0
        self.shift_flash = 0.0
        self.rev_cut = False
        self._prev_up = self._prev_down = False
        # actuators
        self.steer = self.throttle = self.brake = 0.0
        # wheels + lagged weight transfer
        self.omega = np.zeros(4)
        self.dT_long = 0.0
        self.dT_lat = 0.0
        # telemetry
        self.engine_torque = 0.0
        self.engine_power_kw = 0.0
        self.wheel_force = 0.0
        self.wheel_load = np.full(4, self.car.mass * self.sim.gravity / 4)
        self.wheel_slip = np.zeros(4)
        self.wheel_kappa = np.zeros(4)
        self.wheel_grip = np.zeros(4)
        self.slip_f = self.slip_r = 0.0
        self.grip_f = self.grip_r = 0.0
        self.load_f = self.load_r = 0.0
        self.wheelspin = 0.0
        self.aligning_torque = 0.0
        self.handbrake = 0.0
        self.roll = self.pitch = 0.0

    # ------------------------------------------------------------------ #
    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def slip_angle(self):
        return 0.0 if abs(self.vx) < 0.2 else math.atan2(self.vy, self.vx)

    @property
    def lat_g(self):
        return self.ay / self.sim.gravity

    @property
    def long_g(self):
        return self.ax / self.sim.gravity

    @property
    def gear_ratio(self):
        return self.car.gear_ratios[self.gear] * self.car.final_drive

    def _torque_at(self, rpm):
        return float(np.interp(rpm, self._rpm_pts, self._trq_pts))

    # ------------------------------------------------------------------ #
    def step(self, throttle_cmd, brake_cmd, steer_cmd, dt, mu_scale=1.0,
             clutch_cmd=1.0, shift_up=False, shift_down=False,
             manual_gears=False, manual_clutch=False, grade=0.0, handbrake=0.0):
        car, g = self.car, self.sim.gravity
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
        brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))
        clutch_cmd = float(np.clip(clutch_cmd, 0.0, 1.0))

        # ---- human-style rate-limited actuators ----
        mds = car.steer_rate * dt
        self.steer += float(np.clip(steer_cmd * car.max_steer_angle - self.steer, -mds, mds))
        self.throttle += float(np.clip(throttle_cmd - self.throttle,
                                       -car.throttle_rate * dt, car.throttle_rate * dt))
        self.brake += float(np.clip(brake_cmd - self.brake,
                                    -car.brake_rate * dt, car.brake_rate * dt))
        delta = self.steer

        # ---- gearbox: brain (sequential) or auto ----
        self.shift_flash = max(0.0, self.shift_flash - dt)
        if self.shift_timer > 0.0:
            self.shift_timer -= dt
        top = len(car.gear_ratios) - 1
        if manual_gears:
            up_edge = shift_up and not self._prev_up
            down_edge = shift_down and not self._prev_down
            self._prev_up, self._prev_down = shift_up, shift_down
            if self.shift_timer <= 0.0:
                if up_edge and self.gear < top:
                    self.gear += 1; self.shift_timer = car.shift_time; self.shift_flash = 1.0
                elif down_edge and self.gear > 0:
                    self.gear -= 1; self.shift_timer = car.shift_time; self.shift_flash = 1.0
        elif self.shift_timer <= 0.0:
            if self.rpm > car.shift_up_rpm and self.gear < top and self.vx > 1.0:
                self.gear += 1; self.shift_timer = car.shift_time; self.shift_flash = 1.0
            elif self.rpm < car.shift_down_rpm and self.gear > 0:
                self.gear -= 1; self.shift_timer = car.shift_time
        shifting = self.shift_timer > 0.0
        gr = self.gear_ratio

        # ---- clutch engagement ----
        c = clutch_cmd if manual_clutch else 1.0
        if shifting:
            c = 0.0                       # clutch auto-disengages mid-shift
        self.clutch = c

        # ---- turbo boost (slow dynamic; once per body sub-step) ----
        if self.throttle > 0.05 and not shifting:
            spool = (self.rpm - car.boost_spool_start_rpm) / \
                    (car.boost_full_rpm - car.boost_spool_start_rpm)
            boost_target = car.max_boost_bar * float(np.clip(spool, 0, 1)) * self.throttle
            tau = car.boost_spool_tau
        else:
            boost_target, tau = 0.0, car.boost_decay_tau
        self.boost += (boost_target - self.boost) * min(1.0, dt / tau)
        boost_frac = self.boost / car.max_boost_bar if car.max_boost_bar > 0 else 0.0
        torque_mult = car.na_torque_fraction + (1 - car.na_torque_fraction) * boost_frac

        def engine_torque(rpm):
            if rpm >= car.rev_limit_rpm:
                return -car.engine_brake_torque            # rev limiter cut
            if self.throttle < 0.05:
                return -car.engine_brake_torque * (rpm / car.redline_rpm)
            return self._torque_at(rpm) * torque_mult * self.throttle

        # ---- vertical loads with lagged weight transfer ----
        L = car.wheelbase
        base_f = (car.mass * g * car.lr / L) / 2.0
        base_r = (car.mass * g * car.lf / L) / 2.0
        # Longitudinal weight transfer from road gradient:
        # Downhill (negative grade) -> positive transfer -> more front load,
        # less rear load -> promotes oversteer.
        grade_transfer = car.mass * g * (-grade) * car.cg_height / L
        base_f += grade_transfer / 2.0
        base_r -= grade_transfer / 2.0
        long_pw = self.dT_long / 2.0
        lat_f = car.roll_stiffness_front * self.dT_lat
        lat_r = (1.0 - car.roll_stiffness_front) * self.dT_lat
        Fz = np.maximum(np.array([
            base_f - long_pw - lat_f, base_f - long_pw + lat_f,
            base_r + long_pw - lat_r, base_r + long_pw + lat_r]), 0.0)
        self.wheel_load = Fz
        Fz_ref = car.mass * g / 4.0

        # ---- lateral tyre forces + per-wheel constants (computed once) ----
        Rw = car.wheel_radius
        cosd, sind = math.cos(delta), math.sin(delta)
        max_bt = car.max_brake_g * car.mass * g * Rw
        bt_coef = [self.brake * max_bt * (car.brake_bias_front if f else
                   (1 - car.brake_bias_front)) / 2.0 for f in self.is_front]
        vlong = np.zeros(4); cap = np.zeros(4); fy = np.zeros(4)
        for i in range(4):
            vwx = self.vx - self.r * self.wb[i]
            vwy = self.vy + self.r * self.wa[i]
            if self.is_front[i]:
                vlong[i] = vwx * cosd + vwy * sind
                vlat = -vwx * sind + vwy * cosd
            else:
                vlong[i] = vwx; vlat = vwy
            alpha = math.atan2(vlat, abs(vlong[i]) + 0.5)
            self.wheel_slip[i] = alpha
            mu = car.tyre_mu * mu_scale * (1.0 - car.load_sensitivity * (Fz[i] / Fz_ref - 1.0))
            mu = max(0.4 * car.tyre_mu, mu)
            cap[i] = mu * Fz[i]
            fy[i] = -cap[i] * _pac(alpha, car.pacejka_B, car.pacejka_C, car.pacejka_E)

        # ---- stiff drivetrain: engine flywheel + clutch + wheel spin ----
        n_rot = self.sim.drivetrain_substeps
        dt_r = dt / n_rot
        Tc_max, scale, I_e, I_w = (car.clutch_max_torque, car.clutch_slip_scale,
                                   car.engine_inertia, car.wheel_inertia)
        fx = np.zeros(4)
        axle_torque = 0.0
        slip_lock = 8.0                     # rad/s slip below which the clutch grabs
        reflected = 0.5 * I_e * gr * gr     # engine inertia seen at each rear wheel when locked
        hb_rear = handbrake * car.handbrake_torque * 0.5   # extra brake torque per rear wheel
        self.handbrake = handbrake
        for _ in range(n_rot):
            omega_rear = 0.5 * (self.omega[2] + self.omega[3])
            omega_drv = omega_rear * gr
            slip = self.engine_omega - omega_drv
            # Clutch is "locked" when engaged, nearly synchronised, and the
            # driveline is turning at least idle speed; otherwise it slips.
            locked = (c > 0.6 and abs(slip) < slip_lock
                      and omega_drv > self._idle_omega * 0.95)
            rpm_now = (omega_drv if locked else self.engine_omega) * 60.0 / _TWO_PI
            T_eng = engine_torque(rpm_now)
            if locked:
                # engine + driveline turn together and accelerate together
                self.engine_omega = min(omega_drv, self._rev_omega)
                axle_torque = T_eng * gr * car.drivetrain_efficiency
            else:
                # slipping clutch: engine has its own dynamics (free-rev, kick)
                T_clutch = c * Tc_max * math.tanh(slip / scale)
                self.engine_omega += (T_eng - T_clutch) / I_e * dt_r
                self.engine_omega = min(max(self.engine_omega, self._idle_omega), self._rev_omega)
                axle_torque = T_clutch * gr * car.drivetrain_efficiency

            half = 0.5 * axle_torque
            lsd = car.lsd_lock * (self.omega[3] - self.omega[2])
            T_drive = (0.0, 0.0, half + lsd, half - lsd)
            for i in range(4):
                kappa = (self.omega[i] * Rw - vlong[i]) / (abs(vlong[i]) + 0.5)
                fxi = cap[i] * _pac(kappa, car.pacejka_B_long, car.pacejka_C_long, car.pacejka_E_long)
                max_fx = math.sqrt(max(0.0, cap[i] * cap[i] - fy[i] * fy[i]))
                fxi = max(-max_fx, min(max_fx, fxi))
                bt = bt_coef[i] * math.tanh(self.omega[i] / 2.0)
                if i >= 2 and hb_rear > 0.0:        # handbrake locks the rears
                    bt += hb_rear * math.tanh(self.omega[i] / 2.0)
                I_i = I_w + (reflected if (locked and i >= 2) else 0.0)
                self.omega[i] += (T_drive[i] - fxi * Rw - bt) / I_i * dt_r
                self.omega[i] = float(np.clip(self.omega[i], -3.0, 500.0))
                fx[i] = fxi
                self.wheel_kappa[i] = kappa

        self.rpm = float(np.clip(self.engine_omega * 60.0 / _TWO_PI, car.idle_rpm, car.rev_limit_rpm))
        self.rev_cut = self.rpm >= car.rev_limit_rpm
        self.engine_torque = engine_torque(self.rpm)
        self.engine_power_kw = self.engine_torque * self.rpm * _TWO_PI / 60.0 / 1000.0
        self.wheel_force = axle_torque / Rw

        # ---- assemble body forces ----
        Fbx = np.zeros(4); Fby = np.zeros(4); Fy_front = 0.0
        for i in range(4):
            if self.is_front[i]:
                Fbx[i] = fx[i] * cosd - fy[i] * sind
                Fby[i] = fx[i] * sind + fy[i] * cosd
                Fy_front += fy[i]
            else:
                Fbx[i] = fx[i]; Fby[i] = fy[i]
            self.wheel_grip[i] = math.hypot(fx[i], fy[i]) / (cap[i] + 1e-6)

        f_drag = 0.5 * car.air_density * car.drag_cd * car.frontal_area * self.vx * abs(self.vx)
        f_roll = (car.rolling_resistance * car.mass * g * (1.0 if self.vx >= 0 else -1.0)
                  if abs(self.vx) > 0.1 else 0.0)

        Fx_total = float(np.sum(Fbx)) - f_drag - f_roll
        # Gravity component from road gradient:
        # Negative grade = downhill = positive Fx (car accelerates forward).
        Fx_total += -car.mass * g * grade
        Fy_total = float(np.sum(Fby))
        Mz = float(np.sum(self.wa * Fby - self.wb * Fbx))
        self.ax = Fx_total / car.mass
        self.ay = Fy_total / car.mass

        # integrate body
        self.vx += (self.ax + self.vy * self.r) * dt
        self.vy += (self.ay - self.vx * self.r) * dt
        self.r += Mz / car.yaw_inertia * dt
        if self.speed < 0.4 and self.throttle < 0.05 and self.brake > 0.05:
            self.vx *= 0.5; self.vy *= 0.5; self.r *= 0.7; self.omega *= 0.5

        # suspension lag toward new weight-transfer targets
        k = min(1.0, dt / car.suspension_tau)
        self.dT_long += (car.mass * self.ax * car.cg_height / L - self.dT_long) * k
        self.dT_lat += (car.mass * self.ay * car.cg_height / car.track_width - self.dT_lat) * k

        # world pose
        self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
        self.yaw += self.r * dt

        # telemetry roll-ups
        self.load_f = float(Fz[0] + Fz[1]); self.load_r = float(Fz[2] + Fz[3])
        self.grip_f = float(0.5 * (self.wheel_grip[0] + self.wheel_grip[1]))
        self.grip_r = float(0.5 * (self.wheel_grip[2] + self.wheel_grip[3]))
        self.slip_f = float(0.5 * (self.wheel_slip[0] + self.wheel_slip[1]))
        self.slip_r = float(0.5 * (self.wheel_slip[2] + self.wheel_slip[3]))
        self.wheelspin = float(np.clip(max(abs(self.wheel_kappa[2]), abs(self.wheel_kappa[3])),
                                       0.0, 1.5))
        self.aligning_torque = -Fy_front * car.pneumatic_trail
        self.roll = float(np.clip(self.lat_g, -1.5, 1.5)) * car.roll_gain_deg
        self.pitch = float(np.clip(-self.long_g, -1.5, 1.5)) * car.pitch_gain_deg
