"""
6-DOF Rigid Body Dynamics for Insect Flight.

Implements Newton-Euler equations of motion using unit quaternion kinematics
to prevent gimbal lock during agile aerial maneuvers.
Couples left and right wing aerodynamic wrenches (forces and moments) about the body center of mass.
"""
import numpy as np

try:
    from .config import InsectSpec
except (ImportError, ValueError):
    from config import InsectSpec

class InsectBody:
    def __init__(self, spec: InsectSpec):
        self.spec = spec
        self.m = spec.body_mass
        self.I = np.diag([spec.I_xx, spec.I_yy, spec.I_zz])
        self.I_inv = np.diag([1.0 / spec.I_xx, 1.0 / spec.I_yy, 1.0 / spec.I_zz])
        
        # Wing hinge offsets relative to body CoM
        self.r_hinge_L = np.array([spec.hinge_offset_x, spec.hinge_offset_y, spec.hinge_offset_z])
        self.r_hinge_R = np.array([spec.hinge_offset_x, -spec.hinge_offset_y, spec.hinge_offset_z])

        # Body aerodynamic drag (parasitic drag of thorax/abdomen)
        # S_frontal approx pi * r^2
        self.s_frontal = np.pi * (spec.body_radius ** 2)
        self.c_d_body = 0.8 # Blunt ellipsoid drag coefficient

        self.reset()

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None, 
              quat: np.ndarray = None, omega: np.ndarray = None):
        """Resets the state of the insect body."""
        self.pos = np.array(pos if pos is not None else [0.0, 0.0, 0.5], dtype=float) # start at 0.5 m altitude
        self.vel = np.array(vel if vel is not None else [0.0, 0.0, 0.0], dtype=float)
        # Quaternion: [qw, qx, qy, qz]. Default: upright body with nominal pitch
        self.quat = np.array(quat if quat is not None else [1.0, 0.0, 0.0, 0.0], dtype=float)
        self._normalize_quat()
        self.omega = np.array(omega if omega is not None else [0.0, 0.0, 0.0], dtype=float)

    def _normalize_quat(self):
        n = np.linalg.norm(self.quat)
        if n > 1e-8:
            self.quat /= n
        else:
            self.quat = np.array([1.0, 0.0, 0.0, 0.0])

    def rotation_matrix(self) -> np.ndarray:
        """Returns the 3x3 rotation matrix mapping Body -> World coordinates."""
        w, x, y, z = self.quat
        return np.array([
            [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2)]
        ])

    def step(self, F_left: np.ndarray, M_left: np.ndarray,
             F_right: np.ndarray, M_right: np.ndarray, dt: float):
        """
        Integrates body linear and angular equations of motion over time dt (Euler-Maruyama/Verlet).
        """
        R_b2w = self.rotation_matrix()

        # Net body aerodynamic force
        F_aero_body = F_left + F_right
        
        # Net body moment about center of mass:
        # M_total = M_L + M_R + (r_hinge_L x F_L) + (r_hinge_R x F_R)
        M_total = (M_left + M_right + 
                   np.cross(self.r_hinge_L, F_left) + 
                   np.cross(self.r_hinge_R, F_right))

        # Parasitic body drag in world frame
        speed = np.linalg.norm(self.vel)
        if speed > 1e-4:
            F_drag_world = -0.5 * self.spec.air_density * self.s_frontal * self.c_d_body * speed * self.vel
        else:
            F_drag_world = np.zeros(3)

        # 1. Linear dynamics (World frame)
        # F_world = R_b2w @ F_aero_body + F_drag + Gravity
        F_gravity_world = np.array([0.0, 0.0, -self.m * 9.80665])
        F_total_world = R_b2w @ F_aero_body + F_drag_world + F_gravity_world
        acc_world = F_total_world / self.m

        self.vel += acc_world * dt
        self.pos += self.vel * dt

        # Ground collision floor (z >= 0)
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0
            self.vel[2] = max(0.0, self.vel[2])

        # 2. Rotational dynamics (Body frame)
        # I * d_omega/dt = M_total - omega x (I * omega) - damping
        gyro_torque = np.cross(self.omega, self.I @ self.omega)
        # Viscous rotational damping from moving through air
        rot_damping = 1e-7 * self.omega
        alpha_ang = self.I_inv @ (M_total - gyro_torque - rot_damping)

        self.omega += alpha_ang * dt

        # 3. Quaternion integration: q_dot = 0.5 * q (x) [0, omega]
        w, x, y, z = self.quat
        ox, oy, oz = self.omega
        q_dot = 0.5 * np.array([
            -x*ox - y*oy - z*oz,
             w*ox + y*oz - z*oy,
             w*oy - x*oz + z*ox,
             w*oz + x*oy - y*ox
        ])
        self.quat += q_dot * dt
        self._normalize_quat()
