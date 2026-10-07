"""
Vehicle dynamics simulation with nonlinear Pacejka tire friction,
weight transfer, and rear-wheel drive oversteer for drifting.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    """Parameters for a rear-wheel drive drift chassis (e.g. Silvia S15 / BRZ / E46)."""
    # Mass & Geometry
    mass: float = 1250.0            # Total vehicle mass (kg)
    iz: float = 2100.0              # Yaw moment of inertia (kg*m^2)
    lf: float = 1.15                # Distance from CG to front axle (m)
    lr: float = 1.35                # Distance from CG to rear axle (m)
    cg_height: float = 0.45         # Height of Center of Gravity (m)
    wheelbase: float = 2.50         # lf + lr
    width: float = 1.80             # Vehicle width (m)
    length: float = 4.40            # Vehicle length (m)

    # Drivetrain & Brakes
    max_engine_force: float = 5500.0 # Maximum rear drive traction force (N)
    max_brake_force: float = 8000.0  # Maximum service brake force (N)
    front_brake_bias: float = 0.65   # Front brake distribution (65% front, 35% rear)
    max_steer_angle: float = np.radians(45.0) # Maximum steering angle (rad) (~45 deg lock)
    steer_rate: float = np.radians(300.0)     # Steering speed (rad/s)

    # Aerodynamics & Rolling Resistance
    drag_coeff: float = 0.35        # Aerodynamic drag factor (0.5 * rho * Cd * A)
    rolling_resistance: float = 0.015 # Rolling resistance coefficient
    gravity: float = 9.81

    # Tire Friction Parameters (Pacejka Magic Formula)
    # Fy = Fz * D * sin(C * arctan(B*alpha - E*(B*alpha - arctan(B*alpha))))
    tire_mu_peak: float = 1.10      # Peak dry asphalt friction coefficient
    pacejka_B: float = 9.5          # Stiffness factor
    pacejka_C: float = 1.45         # Shape factor
    pacejka_D: float = 1.0          # Peak factor (scaled by mu * Fz)
    pacejka_E: float = -0.15        # Curvature factor


@dataclass
class VehicleState:
    """Current state of the vehicle."""
    x: float = 0.0          # Global X position (m)
    y: float = 0.0          # Global Y position (m)
    yaw: float = 0.0        # Heading yaw angle (rad)
    vx: float = 0.0         # Longitudinal velocity in body frame (m/s)
    vy: float = 0.0         # Lateral velocity in body frame (m/s)
    yaw_rate: float = 0.0   # Yaw angular velocity (rad/s)
    steer: float = 0.0      # Current steering angle (rad)

    @property
    def speed(self) -> float:
        """Total speed in m/s."""
        return float(np.hypot(self.vx, self.vy))

    @property
    def slip_angle(self) -> float:
        """Body slip angle beta = arctan2(vy, vx) in radians."""
        if abs(self.vx) < 0.1:
            return 0.0
        return float(np.arctan2(self.vy, self.vx))

    @property
    def slip_angle_deg(self) -> float:
        """Body slip angle in degrees."""
        return float(np.degrees(self.slip_angle))

    def copy(self) -> "VehicleState":
        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            vx=self.vx,
            vy=self.vy,
            yaw_rate=self.yaw_rate,
            steer=self.steer,
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


class VehicleDynamics:
    """
    Simulates high-fidelity 2D dynamic vehicle response with weight transfer
    and Pacejka friction saturation.
    """

    def __init__(self, params: VehicleParams = None):
        self.params = params or VehicleParams()

    def pacejka_lateral_force(self, slip_angle: float, normal_load: float, mu_scale: float = 1.0) -> float:
        """
        Calculate lateral tire force using Pacejka Magic Formula.
        Positive slip angle produces negative restoring lateral force.
        """
        p = self.params
        mu = p.tire_mu_peak * mu_scale
        D = mu * normal_load
        B = p.pacejka_B
        C = p.pacejka_C
        E = p.pacejka_E

        # Standard sign convention: lateral force opposes tire slip angle
        alpha = -slip_angle
        # Magic formula argument
        bx = B * alpha
        force = D * np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx))))
        return float(force)

    def compute_derivatives(
        self,
        state: VehicleState,
        steer_target: float,
        throttle_brake: float,
        handbrake: float,
    ) -> np.ndarray:
        """
        Compute time derivatives of state: [dx, dy, dyaw, dvx, dvy, dyaw_rate, dsteer].

        Controls:
            steer_target: target steer angle in [-1, 1] (scaled by max_steer_angle)
            throttle_brake: throttle in [0, 1] if positive, footbrake in [-1, 0] if negative
            handbrake: e-brake lever in [0, 1] (locks rear wheels)
        """
        p = self.params
        vx = state.vx
        vy = state.vy
        yaw = state.yaw
        w = state.yaw_rate
        curr_steer = state.steer

        # 1. Steering actuator dynamics (slew-rate limited)
        target_steer_rad = np.clip(steer_target, -1.0, 1.0) * p.max_steer_angle
        steer_diff = target_steer_rad - curr_steer
        steer_deriv = np.clip(steer_diff * 15.0, -p.steer_rate, p.steer_rate)

        # Small speed regularization to prevent singularities at zero speed
        v_blend = np.clip(np.hypot(vx, vy) / 0.5, 0.0, 1.0)
        eff_vx = max(abs(vx), 0.3) * (1.0 if vx >= 0 else -1.0)

        # 2. Tire slip angles
        # Front tire velocity in vehicle frame: (vx, vy + lf * w)
        alpha_f = float(np.arctan2(vy + p.lf * w, abs(eff_vx)) - curr_steer)
        # Rear tire velocity in vehicle frame: (vx, vy - lr * w)
        alpha_r = float(np.arctan2(vy - p.lr * w, abs(eff_vx)))

        # 3. Dynamic normal loads (weight transfer)
        # Approximate longitudinal acceleration from previous step / inputs
        accel_est = throttle_brake * (p.max_engine_force / p.mass) if throttle_brake > 0 else throttle_brake * (p.max_brake_force / p.mass)
        delta_fz = (p.mass * accel_est * p.cg_height) / p.wheelbase
        fz_f_static = p.mass * p.gravity * (p.lr / p.wheelbase)
        fz_r_static = p.mass * p.gravity * (p.lf / p.wheelbase)

        fz_f = np.clip(fz_f_static - delta_fz, 0.1 * fz_f_static, 1.9 * fz_f_static)
        fz_r = np.clip(fz_r_static + delta_fz, 0.1 * fz_r_static, 1.9 * fz_r_static)

        # 4. Longitudinal Forces (drive, brake, handbrake)
        fx_f = 0.0
        fx_r = 0.0

        if throttle_brake >= 0.0:
            # Rear-wheel drive power
            fx_r += throttle_brake * p.max_engine_force
        else:
            # Footbrake distributed front/rear
            total_brake = abs(throttle_brake) * p.max_brake_force
            fx_f -= total_brake * p.front_brake_bias * np.sign(vx) if abs(vx) > 0.1 else 0.0
            fx_r -= total_brake * (1.0 - p.front_brake_bias) * np.sign(vx) if abs(vx) > 0.1 else 0.0

        # Handbrake: forces rear wheels into kinetic lock
        rear_mu_scale = 1.0
        if handbrake > 0.0:
            hb_force = handbrake * (p.tire_mu_peak * fz_r * 0.95)
            fx_r -= hb_force * (np.sign(vx) if abs(vx) > 0.1 else 1.0)
            # Rear lateral traction capacity collapses when wheels lock
            rear_mu_scale = max(0.15, 1.0 - handbrake * 0.75)

        # 5. Friction Ellipse & Lateral Forces
        # Check friction circle capacity for rear tires: Fx^2 + Fy^2 <= (mu * Fz)^2
        max_rear_grip = p.tire_mu_peak * fz_r * rear_mu_scale
        fx_r = np.clip(fx_r, -max_rear_grip * 0.98, max_rear_grip * 0.98)
        remaining_rear_grip_ratio = np.sqrt(max(0.01, 1.0 - (fx_r / max(max_rear_grip, 1.0)) ** 2))

        # Check friction circle capacity for front tires
        max_front_grip = p.tire_mu_peak * fz_f
        fx_f = np.clip(fx_f, -max_front_grip * 0.98, max_front_grip * 0.98)
        remaining_front_grip_ratio = np.sqrt(max(0.01, 1.0 - (fx_f / max(max_front_grip, 1.0)) ** 2))

        # Lateral forces from Pacejka
        fy_f_pure = self.pacejka_lateral_force(alpha_f, fz_f, mu_scale=1.0)
        fy_r_pure = self.pacejka_lateral_force(alpha_r, fz_r, mu_scale=rear_mu_scale)

        fy_f = fy_f_pure * remaining_front_grip_ratio * v_blend
        fy_r = fy_r_pure * remaining_rear_grip_ratio * v_blend

        # Aerodynamic drag and rolling resistance
        f_aero_x = -p.drag_coeff * vx * abs(vx)
        f_roll_x = -p.rolling_resistance * p.mass * p.gravity * np.sign(vx) if abs(vx) > 0.1 else 0.0

        # 6. Equations of Motion in Car Body Frame
        # Front wheel force rotated by steer angle curr_steer
        cos_steer = np.cos(curr_steer)
        sin_steer = np.sin(curr_steer)

        fx_total = (fx_f * cos_steer - fy_f * sin_steer) + fx_r + f_aero_x + f_roll_x
        fy_total = (fx_f * sin_steer + fy_f * cos_steer) + fy_r

        # Acceleration in body frame:
        # dvx/dt = (Fx / m) + vy * w
        # dvy/dt = (Fy / m) - vx * w
        # dw/dt  = (lf * (fx_f * sin_steer + fy_f * cos_steer) - lr * fy_r) / Iz
        dvx = (fx_total / p.mass) + vy * w
        dvy = (fy_total / p.mass) - vx * w

        torque_front = p.lf * (fx_f * sin_steer + fy_f * cos_steer)
        torque_rear = -p.lr * fy_r
        # Small yaw damping to stabilize high-frequency numerical oscillation
        yaw_damping = -250.0 * w
        dw = (torque_front + torque_rear + yaw_damping) / p.iz

        # Stop drift jitter when car is virtually stopped and no throttle
        if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(throttle_brake) < 0.05 and handbrake < 0.05:
            dvx = -vx * 10.0
            dvy = -vy * 10.0
            dw = -w * 10.0

        # 7. Global Kinematics
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
        dt: float = 0.02,
        substeps: int = 2,
    ) -> VehicleState:
        """
        Integrate vehicle state forward in time using Runge-Kutta 4th Order
        with optional internal sub-stepping for ultra-stable drift dynamics.
        """
        sub_dt = dt / float(substeps)
        s_arr = state.to_array()

        for _ in range(substeps):
            st = VehicleState.from_array(s_arr)
            k1 = self.compute_derivatives(st, steer, throttle_brake, handbrake)

            st2 = VehicleState.from_array(s_arr + 0.5 * sub_dt * k1)
            k2 = self.compute_derivatives(st2, steer, throttle_brake, handbrake)

            st3 = VehicleState.from_array(s_arr + 0.5 * sub_dt * k2)
            k3 = self.compute_derivatives(st3, steer, throttle_brake, handbrake)

            st4 = VehicleState.from_array(s_arr + sub_dt * k3)
            k4 = self.compute_derivatives(st4, steer, throttle_brake, handbrake)

            s_arr += (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Normalize yaw angle to [-pi, pi]
            s_arr[2] = (s_arr[2] + np.pi) % (2.0 * np.pi) - np.pi
            # Clamp steering angle
            s_arr[6] = np.clip(s_arr[6], -self.params.max_steer_angle, self.params.max_steer_angle)

        return VehicleState.from_array(s_arr)

    def get_car_corners(self, state: VehicleState) -> np.ndarray:
        """
        Return the 4 corner coordinates of the car bounding box in world space.
        Returns shape (4, 2): [front_left, front_right, rear_right, rear_left].
        """
        hw = self.params.width * 0.5
        hl_f = self.params.lf + 0.6  # front bumper
        hl_r = self.params.lr + 0.6  # rear bumper

        # Local body corners: (x_forward, y_left)
        corners_local = np.array([
            [hl_f,  hw],   # Front Left
            [hl_f, -hw],   # Front Right
            [-hl_r, -hw],  # Rear Right
            [-hl_r,  hw],  # Rear Left
        ], dtype=np.float64)

        cos_y = np.cos(state.yaw)
        sin_y = np.sin(state.yaw)
        rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)

        corners_world = (corners_local @ rot_mat.T) + np.array([state.x, state.y])
        return corners_world
