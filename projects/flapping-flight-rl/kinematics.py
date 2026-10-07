"""
Insect Wing Kinematics and Geometry.

Models 3-DOF articulation per wing:
  - phi (stroke / sweep angle in the stroke plane)
  - theta (deviation / elevation angle out of the stroke plane)
  - psi (pitch angle about the spanwise axis, setting angle of attack alpha)

Tracks dorsal wing-tip separation to detect the Weis-Fogh 'clap-and-fling' mechanism.
"""
from dataclasses import dataclass
import numpy as np

try:
    from .config import InsectSpec
except (ImportError, ValueError):
    from config import InsectSpec

@dataclass
class WingState:
    phi: float = 0.0          # Stroke angle (rad)
    theta: float = 0.0        # Elevation/deviation angle (rad)
    psi: float = 0.0          # Wing pitch angle (rad)
    
    phi_dot: float = 0.0      # Angular velocities (rad/s)
    theta_dot: float = 0.0
    psi_dot: float = 0.0

    phi_ddot: float = 0.0     # Angular accelerations (rad/s^2)
    theta_ddot: float = 0.0
    psi_ddot: float = 0.0

    tip_pos_body: np.ndarray = None # 3D tip position in body frame

class KinematicEngine:
    def __init__(self, spec: InsectSpec):
        self.spec = spec
        self.R = spec.wing_length
        self.stroke_plane_angle = spec.stroke_plane_angle
        
        # Left and Right wing hinge positions in body frame (x=forward, y=left, z=up)
        self.hinge_left = np.array([spec.hinge_offset_x, spec.hinge_offset_y, spec.hinge_offset_z])
        self.hinge_right = np.array([spec.hinge_offset_x, -spec.hinge_offset_y, spec.hinge_offset_z])

        # Clap-and-fling tracking state
        self.is_clapping = False
        self.fling_boost_timer = 0.0

    def compute_wing_angles(self, t: float, 
                             bias_phi_L: float = 0.0, bias_phi_R: float = 0.0,
                             pitch_offset_L: float = 0.0, pitch_offset_R: float = 0.0,
                             elevation_offset_L: float = 0.0, elevation_offset_R: float = 0.0,
                             freq_offset: float = 0.0) -> tuple[WingState, WingState]:
        """
        Computes analytical stroke kinematics at time t with additive policy offsets.
        """
        f = max(5.0, self.spec.nominal_frequency + freq_offset)
        omega = 2.0 * np.pi * f
        phase = omega * t
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        half_amp = self.spec.stroke_amplitude / 2.0
        mean_bias = getattr(self.spec, "stroke_mean_bias", 0.0)

        # --- Left Wing ---
        phi_L = mean_bias + half_amp * cos_p + bias_phi_L
        phi_dot_L = -half_amp * omega * sin_p
        phi_ddot_L = -half_amp * (omega ** 2) * cos_p

        # Nominal figure-8 deviation
        theta_L = np.radians(8.0) * np.sin(2.0 * phase) + elevation_offset_L
        theta_dot_L = 2.0 * omega * np.radians(8.0) * np.cos(2.0 * phase)
        theta_ddot_L = -4.0 * (omega ** 2) * np.radians(8.0) * np.sin(2.0 * phase)

        # Dynamic pitch (smooth trapezoidal wave between downstroke and upstroke AoA)
        pitch_nominal_L = np.radians(45.0) * np.tanh(3.0 * np.sin(phase))
        psi_L = pitch_nominal_L + pitch_offset_L
        psi_dot_L = np.radians(45.0) * 3.0 * omega * np.cos(phase) * (1.0 - np.tanh(3.0 * np.sin(phase)) ** 2)
        psi_ddot_L = 0.0

        left_state = WingState(
            phi=phi_L, theta=theta_L, psi=psi_L,
            phi_dot=phi_dot_L, theta_dot=theta_dot_L, psi_dot=psi_dot_L,
            phi_ddot=phi_ddot_L, theta_ddot=theta_ddot_L, psi_ddot=psi_ddot_L
        )

        # --- Right Wing (mirrored lateral sign) ---
        phi_R = mean_bias + half_amp * cos_p + bias_phi_R
        phi_dot_R = -half_amp * omega * sin_p
        phi_ddot_R = -half_amp * (omega ** 2) * cos_p

        theta_R = np.radians(8.0) * np.sin(2.0 * phase) + elevation_offset_R
        theta_dot_R = 2.0 * omega * np.radians(8.0) * np.cos(2.0 * phase)
        theta_ddot_R = -4.0 * (omega ** 2) * np.radians(8.0) * np.sin(2.0 * phase)

        pitch_nominal_R = np.radians(45.0) * np.tanh(3.0 * np.sin(phase))
        psi_R = pitch_nominal_R + pitch_offset_R
        psi_dot_R = np.radians(45.0) * 3.0 * omega * np.cos(phase) * (1.0 - np.tanh(3.0 * np.sin(phase)) ** 2)
        psi_ddot_R = 0.0

        right_state = WingState(
            phi=phi_R, theta=theta_R, psi=psi_R,
            phi_dot=phi_dot_R, theta_dot=theta_dot_R, psi_dot=psi_dot_R,
            phi_ddot=phi_ddot_R, theta_ddot=theta_ddot_R, psi_ddot=psi_ddot_R
        )

        # Calculate wingtip locations to check clap-and-fling
        self._update_tip_positions(left_state, right_state)
        self._check_clap_and_fling(left_state, right_state, f)

        return left_state, right_state

    def _update_tip_positions(self, left: WingState, right: WingState):
        """Calculates 3D wing-tip positions in the body frame."""
        beta = self.stroke_plane_angle
        cos_b, sin_b = np.cos(beta), np.sin(beta)

        for state, hinge, is_left in [(left, self.hinge_left, True), (right, self.hinge_right, False)]:
            sign_y = 1.0 if is_left else -1.0
            r_sp = np.array([
                -self.R * np.cos(state.theta) * np.sin(state.phi),
                sign_y * self.R * np.cos(state.theta) * np.cos(state.phi),
                self.R * np.sin(state.theta)
            ])
            R_sp2b = np.array([
                [cos_b, 0.0, sin_b],
                [0.0,   1.0, 0.0],
                [-sin_b, 0.0, cos_b]
            ])
            state.tip_pos_body = hinge + R_sp2b @ r_sp

    def _check_clap_and_fling(self, left: WingState, right: WingState, f: float):
        """Monitors inter-wing tip distance to detect clap and fling events."""
        dist = np.linalg.norm(left.tip_pos_body - right.tip_pos_body)
        
        # During dorsal stroke reversal (phi near maximum stroke angle)
        at_dorsal = (left.phi > 0.6 * (self.spec.stroke_amplitude / 2.0))
        
        if at_dorsal and dist < self.spec.clap_distance_threshold:
            self.is_clapping = True
        elif self.is_clapping and (left.phi_dot < 0): # Beginning of downstroke = Fling
            self.is_clapping = False
            # Fling boosts circulation for the first ~25% of the stroke
            self.fling_boost_timer = 0.25 / f
