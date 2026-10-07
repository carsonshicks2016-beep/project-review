"""
Tandem Vehicle Dynamics Simulation.
Features:
- Nonlinear Pacejka Magic Formula tire friction with dynamic weight transfer.
- Aerodynamic wake cone ('dirty air') shed by the leader car:
  - Downforce loss causing front axle washout/understeer.
  - Stochastic vortex-shedding buffeting forces and yaw moments.
- Dynamic tire smoke friction degradation (rubber and smoke cloud reducing local grip).
- Separating Axis Theorem (SAT) oriented bounding box collision detection and response.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VehicleParams:
    """Parameters for rear-wheel drive tandem drift chassis (e.g. Silvia S15 / BRZ / E46)."""
    # Mass & Geometry
    mass: float = 1250.0             # Vehicle curb mass (kg)
    iz: float = 2100.0               # Yaw moment of inertia (kg*m^2)
    lf: float = 1.15                 # CG to front axle (m)
    lr: float = 1.35                 # CG to rear axle (m)
    cg_height: float = 0.45          # CG height (m)
    wheelbase: float = 2.50          # lf + lr (m)
    width: float = 1.80              # Overall width (m)
    length: float = 4.40             # Overall length (m)

    # Drivetrain & Actuation
    max_engine_force: float = 5800.0 # Maximum rear drive traction force (N)
    max_brake_force: float = 8500.0  # Maximum service brake force (N)
    front_brake_bias: float = 0.65    # Front brake distribution (65% front)
    max_steer_angle: float = np.radians(48.0) # Maximum steer angle (~48 deg lock)
    steer_rate: float = np.radians(320.0)      # Steering angular speed (rad/s)

    # Aerodynamics & Rolling Resistance
    drag_coeff: float = 0.36         # 0.5 * rho * Cd * A
    downforce_f_coeff: float = 0.25  # Front aero downforce factor (0.5 * rho * Cl_f * A)
    downforce_r_coeff: float = 0.40  # Rear aero downforce factor (0.5 * rho * Cl_r * A)
    rolling_resistance: float = 0.015
    gravity: float = 9.81

    # Pacejka Tire Model (Fy = Fz * D * sin(C * arctan(B*alpha - E*(B*alpha - arctan(B*alpha)))))
    tire_mu_peak: float = 1.12       # Peak asphalt friction
    pacejka_B: float = 9.8           # Stiffness factor
    pacejka_C: float = 1.45          # Shape factor
    pacejka_D: float = 1.0           # Peak factor
    pacejka_E: float = -0.15         # Curvature factor

    # Aerodynamic Wake Parameters
    wake_spread_angle: float = np.radians(14.0) # Wake cone expansion half-angle
    wake_length: float = 22.0                   # Wake decay distance (m)
    wake_max_deficit: float = 0.55              # Max dynamic pressure loss in wake center
    wake_buffet_force: float = 380.0            # Max lateral buffeting force (N)
    wake_buffet_torque: float = 450.0           # Max yaw buffeting torque (Nm)


@dataclass
class VehicleState:
    """State vector of a single drift car."""
    x: float = 0.0          # Global X coordinate (m)
    y: float = 0.0          # Global Y coordinate (m)
    yaw: float = 0.0        # Heading angle (rad, 0 along +X)
    vx: float = 0.0         # Longitudinal velocity in body frame (m/s)
    vy: float = 0.0         # Lateral velocity in body frame (m/s)
    yaw_rate: float = 0.0   # Yaw rate d(yaw)/dt (rad/s)
    steer: float = 0.0      # Current steer angle (rad)

    # Telemetry and turbulence flags
    in_wake: float = 0.0    # Normalized wake intensity experienced [0, 1]
    smoke_exposure: float = 0.0 # Normalized smoke density experienced [0, 1]

    @property
    def speed(self) -> float:
        """Total planar speed (m/s)."""
        return float(np.hypot(self.vx, self.vy))

    @property
    def slip_angle(self) -> float:
        """Body slip angle beta = arctan2(vy, vx) in radians."""
        if abs(self.vx) < 0.2:
            return 0.0
        return float(np.arctan2(self.vy, self.vx))

    @property
    def slip_angle_deg(self) -> float:
        """Body slip angle in degrees."""
        return float(np.degrees(self.slip_angle))

    @property
    def velocity_world(self) -> np.ndarray:
        """Velocity vector in world coordinates [vx_w, vy_w]."""
        cos_y = np.cos(self.yaw)
        sin_y = np.sin(self.yaw)
        return np.array([
            self.vx * cos_y - self.vy * sin_y,
            self.vx * sin_y + self.vy * cos_y
        ], dtype=np.float64)

    def copy(self) -> "VehicleState":
        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            vx=self.vx,
            vy=self.vy,
            yaw_rate=self.yaw_rate,
            steer=self.steer,
            in_wake=self.in_wake,
            smoke_exposure=self.smoke_exposure,
        )

    def to_array(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.yaw, self.vx, self.vy, self.yaw_rate, self.steer
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VehicleState":
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            yaw=float(arr[2]),
            vx=float(arr[3]),
            vy=float(arr[4]),
            yaw_rate=float(arr[5]),
            steer=float(arr[6]),
        )


@dataclass
class WakeField:
    """Represents the aerodynamic dirty air and smoke plume cast by the leader."""
    leader_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    leader_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    leader_yaw: float = 0.0
    intensity: float = 0.0
    smoke_clouds: List[Tuple[float, float, float, float]] = field(default_factory=list) # (x, y, radius, density)


class TandemDynamics:
    """
    Simulates high-fidelity multi-vehicle tandem drift physics with
    aerodynamic wake cone coupling, dirty air washout, and mutual collision resolution.
    """

    def __init__(self, params: Optional[VehicleParams] = None):
        self.params = params or VehicleParams()

    def pacejka_lateral_force(self, slip_angle: float, normal_load: float, mu_scale: float = 1.0) -> float:
        """
        Pacejka Magic Formula lateral tire force.
        Positive slip angle generates restoring lateral force opposing tire slip.
        """
        p = self.params
        mu = p.tire_mu_peak * mu_scale
        D = mu * normal_load
        B = p.pacejka_B
        C = p.pacejka_C
        E = p.pacejka_E

        alpha = -slip_angle
        bx = B * alpha
        force = D * np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx))))
        return float(force)

    def compute_wake_interaction(
        self,
        leader: VehicleState,
        follower: VehicleState,
    ) -> Tuple[float, float, float]:
        """
        Calculates wake intensity, dirty air downforce loss, and buffeting moments
        experienced by the follower car behind the leader.

        Returns:
            wake_intensity: [0.0, 1.0] intensity of dirty air exposure
            buffet_fy: Stochastic lateral buffeting force (N)
            buffet_torque: Stochastic yaw buffeting moment (Nm)
        """
        p = self.params
        lead_vel = leader.velocity_world
        lead_speed = float(np.linalg.norm(lead_vel))

        if lead_speed < 2.0:
            return 0.0, 0.0, 0.0

        # Wake sheds along the opposite direction of the leader's ground velocity vector
        wake_dir = -lead_vel / lead_speed  # Unit vector pointing backwards into wake
        wake_normal = np.array([-wake_dir[1], wake_dir[0]])

        # Displacement vector from leader CG to follower CG
        rel_pos = np.array([follower.x - leader.x, follower.y - leader.y])

        # Distance downstream into the wake cone
        d_downstream = float(np.dot(rel_pos, wake_dir))
        # Lateral distance from the wake centerline
        d_cross = float(abs(np.dot(rel_pos, wake_normal)))

        # If follower is ahead of leader or outside maximum wake length
        if d_downstream <= 0.5 or d_downstream > p.wake_length:
            return 0.0, 0.0, 0.0

        # Wake cone half-width at downstream distance d_downstream
        half_width = (p.width * 0.5) + d_downstream * np.tan(p.wake_spread_angle)

        if d_cross > half_width:
            return 0.0, 0.0, 0.0

        # Normalized wake intensity: 1.0 at leader bumper on centerline, decaying with distance and width
        cross_factor = 1.0 - (d_cross / half_width) ** 2
        dist_factor = np.exp(-1.8 * (d_downstream / p.wake_length))
        speed_factor = np.clip(lead_speed / 20.0, 0.2, 1.0)

        wake_intensity = float(cross_factor * dist_factor * speed_factor)
        wake_intensity = np.clip(wake_intensity, 0.0, 1.0)

        # Stochastic vortex shedding buffeting forces (frequency increases with speed)
        # Using pseudo-random turbulence based on relative positions
        phase = d_downstream * 2.5 + follower.x * 0.3
        turb_noise_fy = np.sin(phase * 7.1) + 0.5 * np.cos(phase * 13.7)
        turb_noise_mz = np.cos(phase * 5.3) - 0.4 * np.sin(phase * 11.2)

        buffet_fy = float(wake_intensity * p.wake_buffet_force * turb_noise_fy)
        buffet_torque = float(wake_intensity * p.wake_buffet_torque * turb_noise_mz)

        return wake_intensity, buffet_fy, buffet_torque

    def compute_smoke_friction_loss(
        self,
        car: VehicleState,
        smoke_field: List[Tuple[float, float, float, float]],
    ) -> float:
        """
        Computes local road friction reduction factor [0, 1] due to tire smoke and rubber marbles.
        """
        if not smoke_field:
            return 1.0

        car_pos = np.array([car.x, car.y])
        total_density = 0.0

        for sx, sy, srad, sdensity in smoke_field:
            dist = float(np.hypot(car_pos[0] - sx, car_pos[1] - sy))
            if dist < srad:
                total_density += sdensity * (1.0 - dist / srad)

        smoke_factor = np.clip(total_density * 0.35, 0.0, 0.30)
        return float(1.0 - smoke_factor)

    def compute_derivatives(
        self,
        state: VehicleState,
        steer_target: float,
        throttle_brake: float,
        handbrake: float,
        wake_intensity: float = 0.0,
        buffet_fy: float = 0.0,
        buffet_torque: float = 0.0,
        friction_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Computes state derivatives [dx, dy, dyaw, dvx, dvy, dyaw_rate, dsteer].
        Incorporates wake turbulence downforce loss on front axle (inducing understeer)
        and aerodynamic buffeting.
        """
        p = self.params
        vx = state.vx
        vy = state.vy
        yaw = state.yaw
        w = state.yaw_rate
        curr_steer = state.steer

        # 1. Steering actuator dynamics
        target_steer_rad = np.clip(steer_target, -1.0, 1.0) * p.max_steer_angle
        steer_diff = target_steer_rad - curr_steer
        steer_deriv = np.clip(steer_diff * 16.0, -p.steer_rate, p.steer_rate)

        # Speed conditioning
        v_speed = np.hypot(vx, vy)
        v_blend = np.clip(v_speed / 0.5, 0.0, 1.0)
        eff_vx = max(abs(vx), 0.3) * (1.0 if vx >= 0 else -1.0)

        # 2. Tire slip angles
        alpha_f = float(np.arctan2(vy + p.lf * w, abs(eff_vx)) - curr_steer)
        alpha_r = float(np.arctan2(vy - p.lr * w, abs(eff_vx)))

        # 3. Dynamic normal loads (weight transfer + aero downforce)
        accel_est = (
            throttle_brake * (p.max_engine_force / p.mass)
            if throttle_brake > 0
            else throttle_brake * (p.max_brake_force / p.mass)
        )
        delta_fz_accel = (p.mass * accel_est * p.cg_height) / p.wheelbase
        fz_f_static = p.mass * p.gravity * (p.lr / p.wheelbase)
        fz_r_static = p.mass * p.gravity * (p.lf / p.wheelbase)

        # Aerodynamic downforce: speed squared
        # Dirty air reduces front downforce drastically (up to 60% loss in full wake)
        front_aero_loss = 1.0 - 0.60 * wake_intensity
        rear_aero_loss = 1.0 - 0.20 * wake_intensity

        f_down_f = p.downforce_f_coeff * (vx ** 2) * front_aero_loss
        f_down_r = p.downforce_r_coeff * (vx ** 2) * rear_aero_loss

        fz_f = np.clip(fz_f_static - delta_fz_accel + f_down_f, 0.1 * fz_f_static, 2.2 * fz_f_static)
        fz_r = np.clip(fz_r_static + delta_fz_accel + f_down_r, 0.1 * fz_r_static, 2.2 * fz_r_static)

        # 4. Longitudinal Forces
        fx_f = 0.0
        fx_r = 0.0

        if throttle_brake >= 0.0:
            fx_r += throttle_brake * p.max_engine_force
        else:
            total_brake = abs(throttle_brake) * p.max_brake_force
            sign_vx = np.sign(vx) if abs(vx) > 0.1 else 0.0
            fx_f -= total_brake * p.front_brake_bias * sign_vx
            fx_r -= total_brake * (1.0 - p.front_brake_bias) * sign_vx

        # Handbrake rear axle lock
        rear_mu_scale = friction_scale
        if handbrake > 0.0:
            hb_force = handbrake * (p.tire_mu_peak * fz_r * 0.95 * friction_scale)
            fx_r -= hb_force * (np.sign(vx) if abs(vx) > 0.1 else 1.0)
            rear_mu_scale = max(0.12, friction_scale * (1.0 - handbrake * 0.78))

        # 5. Friction Ellipse & Lateral Forces
        max_rear_grip = p.tire_mu_peak * fz_r * rear_mu_scale
        fx_r = np.clip(fx_r, -max_rear_grip * 0.98, max_rear_grip * 0.98)
        rem_rear_grip = np.sqrt(max(0.01, 1.0 - (fx_r / max(max_rear_grip, 1.0)) ** 2))

        front_mu_scale = friction_scale
        max_front_grip = p.tire_mu_peak * fz_f * front_mu_scale
        fx_f = np.clip(fx_f, -max_front_grip * 0.98, max_front_grip * 0.98)
        rem_front_grip = np.sqrt(max(0.01, 1.0 - (fx_f / max(max_front_grip, 1.0)) ** 2))

        # Lateral Pacejka forces
        fy_f_pure = self.pacejka_lateral_force(alpha_f, fz_f, mu_scale=front_mu_scale)
        fy_r_pure = self.pacejka_lateral_force(alpha_r, fz_r, mu_scale=rear_mu_scale)

        fy_f = fy_f_pure * rem_front_grip * v_blend
        fy_r = fy_r_pure * rem_rear_grip * v_blend

        # Aero drag & rolling resistance
        # In wake, overall drag is slightly reduced (slipstreaming)
        drag_scale = 1.0 - 0.35 * wake_intensity
        f_aero_x = -p.drag_coeff * drag_scale * vx * abs(vx)
        f_roll_x = -p.rolling_resistance * p.mass * p.gravity * (np.sign(vx) if abs(vx) > 0.1 else 0.0)

        # 6. Equations of Motion in Body Frame
        cos_steer = np.cos(curr_steer)
        sin_steer = np.sin(curr_steer)

        fx_total = (fx_f * cos_steer - fy_f * sin_steer) + fx_r + f_aero_x + f_roll_x
        # Add wake buffeting forces
        fy_total = (fx_f * sin_steer + fy_f * cos_steer) + fy_r + buffet_fy

        dvx = (fx_total / p.mass) + vy * w
        dvy = (fy_total / p.mass) - vx * w

        torque_front = p.lf * (fx_f * sin_steer + fy_f * cos_steer)
        torque_rear = -p.lr * fy_r
        yaw_damping = -260.0 * w
        dw = (torque_front + torque_rear + yaw_damping + buffet_torque) / p.iz

        # World velocity derivatives
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dx = vx * cos_yaw - vy * sin_yaw
        dy = vx * sin_yaw + vy * cos_yaw
        dyaw = w

        return np.array([dx, dy, dyaw, dvx, dvy, dw, steer_deriv], dtype=np.float64)

    def step_rk4(
        self,
        state: VehicleState,
        steer: float,
        throttle_brake: float,
        handbrake: float,
        wake_intensity: float = 0.0,
        buffet_fy: float = 0.0,
        buffet_torque: float = 0.0,
        friction_scale: float = 1.0,
        dt: float = 0.02,
    ) -> VehicleState:
        """4th-order Runge-Kutta numerical integration step."""
        y0 = state.to_array()

        k1 = self.compute_derivatives(
            VehicleState.from_array(y0),
            steer, throttle_brake, handbrake,
            wake_intensity, buffet_fy, buffet_torque, friction_scale
        )
        k2 = self.compute_derivatives(
            VehicleState.from_array(y0 + 0.5 * dt * k1),
            steer, throttle_brake, handbrake,
            wake_intensity, buffet_fy, buffet_torque, friction_scale
        )
        k3 = self.compute_derivatives(
            VehicleState.from_array(y0 + 0.5 * dt * k2),
            steer, throttle_brake, handbrake,
            wake_intensity, buffet_fy, buffet_torque, friction_scale
        )
        k4 = self.compute_derivatives(
            VehicleState.from_array(y0 + dt * k3),
            steer, throttle_brake, handbrake,
            wake_intensity, buffet_fy, buffet_torque, friction_scale
        )

        y_next = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Normalize yaw to [-pi, pi]
        y_next[2] = (y_next[2] + np.pi) % (2.0 * np.pi) - np.pi

        next_state = VehicleState.from_array(y_next)
        next_state.in_wake = wake_intensity
        next_state.smoke_exposure = float(1.0 - friction_scale) / 0.30 if friction_scale < 1.0 else 0.0
        return next_state

    def get_car_corners(self, state: VehicleState) -> np.ndarray:
        """
        Returns the 4 corner points of the vehicle in world coordinates.
        Shape (4, 2): [front_left, front_right, rear_right, rear_left]
        """
        p = self.params
        hw = p.width * 0.5
        hl_f = p.lf + 0.95  # Front bumper overhang
        hl_r = p.lr + 0.95  # Rear bumper overhang

        corners_body = np.array([
            [hl_f, hw],
            [hl_f, -hw],
            [-hl_r, -hw],
            [-hl_r, hw],
        ], dtype=np.float64)

        cos_y = np.cos(state.yaw)
        sin_y = np.sin(state.yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)

        return (corners_body @ rot.T) + np.array([state.x, state.y])

    def check_car_collision_sat(
        self,
        car1_corners: np.ndarray,
        car2_corners: np.ndarray,
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Separating Axis Theorem (SAT) oriented bounding box collision test.
        Returns:
            colliding: bool
            penetration_depth: float (minimum overlap along collision normal)
            collision_normal: np.ndarray shape (2,)
        """
        def get_axes(corners):
            axes = []
            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    axes.append(normal / norm)
            return axes

        axes = get_axes(car1_corners) + get_axes(car2_corners)
        min_overlap = float("inf")
        best_axis = np.zeros(2)

        for axis in axes:
            proj1 = car1_corners @ axis
            proj2 = car2_corners @ axis

            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            overlap = min(max1, max2) - max(min1, min2)
            if overlap <= 0:
                return False, 0.0, np.zeros(2)

            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = axis

        # Direct normal from car1 to car2
        c1 = np.mean(car1_corners, axis=0)
        c2 = np.mean(car2_corners, axis=0)
        if np.dot(c2 - c1, best_axis) < 0:
            best_axis = -best_axis

        return True, float(min_overlap), best_axis

    def resolve_elastic_collision(
        self,
        car1: VehicleState,
        car2: VehicleState,
        normal: np.ndarray,
        penetration: float,
        restitution: float = 0.45,
    ) -> Tuple[VehicleState, VehicleState]:
        """
        Applies impulse-based momentum transfer to separate colliding cars cleanly.
        """
        p = self.params
        c1 = car1.copy()
        c2 = car2.copy()

        # Positional separation
        sep = normal * (penetration * 0.55)
        c1.x -= sep[0]
        c1.y -= sep[1]
        c2.x += sep[0]
        c2.y += sep[1]

        # World velocities
        v1_w = c1.velocity_world
        v2_w = c2.velocity_world
        v_rel = v2_w - v1_w

        vn = float(np.dot(v_rel, normal))
        if vn < 0:  # Moving toward each other
            impulse_mag = -(1.0 + restitution) * vn * (p.mass * 0.5)
            impulse_vec = impulse_mag * normal

            v1_new_w = v1_w - (impulse_vec / p.mass)
            v2_new_w = v2_w + (impulse_vec / p.mass)

            # Convert back to car body frame
            def to_body(v_w, yaw):
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                return (
                    v_w[0] * cos_y + v_w[1] * sin_y,
                    -v_w[0] * sin_y + v_w[1] * cos_y
                )

            c1.vx, c1.vy = to_body(v1_new_w, c1.yaw)
            c2.vx, c2.vy = to_body(v2_new_w, c2.yaw)

        return c1, c2
