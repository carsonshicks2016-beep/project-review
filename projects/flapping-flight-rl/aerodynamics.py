"""
Unsteady Aerodynamic Solver for Flapping Insect Flight.

Implements the Dickinson-Sane blade-element model:
  1. Translational Lift & Drag with dynamic Leading-Edge Vortex (LEV) attachment
  2. Rotational circulation (Kramer effect) proportional to chord-pitch rate
  3. Added-mass virtual fluid acceleration normal to the chord
  4. Weis-Fogh 'Clap-and-Fling' circulation boost upon dorsal stroke reversal

Integrates spanwise strip forces into total wing aerodynamic wrench (F, M).
"""
import numpy as np

try:
    from .config import InsectSpec
    from .kinematics import WingState
except (ImportError, ValueError):
    from config import InsectSpec
    from kinematics import WingState

class AerodynamicSolver:
    def __init__(self, spec: InsectSpec):
        self.spec = spec
        self.N = spec.blade_elements
        self.dr = spec.wing_length / self.N
        
        # Radii of blade elements along span
        self.r_strips = np.linspace(self.dr / 2.0, spec.wing_length - self.dr / 2.0, self.N)
        
        # Chord distribution (standard biological Betz/elliptic approximation)
        r_norm = self.r_strips / spec.wing_length
        self.chord_strips = (4.0 * spec.mean_chord / np.pi) * np.sqrt(np.maximum(1e-4, 1.0 - r_norm**2))
        self.area_strips = self.chord_strips * self.dr

    def compute_wing_wrench(self, state: WingState, is_left: bool, 
                            fling_active: bool, body_vel: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Computes the net aerodynamic force (F) and torque (M) about the hinge in body coordinates.
        """
        if body_vel is None:
            body_vel = np.zeros(3)

        circ_mult = self.spec.clap_boost_factor if fling_active else 1.0

        beta = self.spec.stroke_plane_angle
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        # Rotation from Stroke Plane to Body Frame
        R_sp2b = np.array([
            [cos_b, 0.0, sin_b],
            [0.0,   1.0, 0.0],
            [-sin_b, 0.0, cos_b]
        ])

        sign_y = 1.0 if is_left else -1.0
        cp, sp = np.cos(state.phi), np.sin(state.phi)
        ct, st = np.cos(state.theta), np.sin(state.theta)

        # Span unit vector in stroke plane
        span_sp = np.array([-ct * sp, sign_y * ct * cp, st])
        norm_span = np.linalg.norm(span_sp)
        if norm_span > 1e-6:
            span_sp /= norm_span

        total_F_sp = np.zeros(3)
        total_M_sp = np.zeros(3)

        total_lift = 0.0
        total_drag = 0.0
        total_rot = 0.0
        total_added_mass = 0.0

        for r, c, dS in zip(self.r_strips, self.chord_strips, self.area_strips):
            # Strip position in stroke plane
            r_elem = r * span_sp

            # Linear flapping velocity of strip in stroke plane
            vx_sp = -r * (ct * cp * state.phi_dot - st * sp * state.theta_dot)
            vy_sp = sign_y * r * (-ct * sp * state.phi_dot - st * cp * state.theta_dot)
            vz_sp = r * ct * state.theta_dot
            v_wing_sp = np.array([vx_sp, vy_sp, vz_sp])

            # Relative flow velocity seen by the wing
            v_rel = -v_wing_sp
            # Inflow normal to span (chordwise plane)
            v_chord = v_rel - np.dot(v_rel, span_sp) * span_sp
            speed = np.linalg.norm(v_chord)

            if speed < 1e-4:
                continue

            # Unit drag direction (opposes chordwise motion)
            e_D = v_chord / speed

            # Unit lift direction (perpendicular to relative velocity and span, pointing upward)
            if is_left:
                e_L = np.cross(span_sp, e_D)
            else:
                e_L = np.cross(e_D, span_sp)
            
            # The wing's aerodynamic suction side faces dorsal (+z_sp) on both half-strokes
            if e_L[2] < 0:
                e_L = -e_L

            # Dynamic Angle of Attack alpha (Dickinson/Sane model)
            # Effective alpha peaks at ~45 degrees during mid-stroke
            alpha = np.radians(45.0) * (abs(state.phi_dot) / (max(1e-3, (self.spec.stroke_amplitude / 2.0) * (2.0 * np.pi * self.spec.nominal_frequency))))
            alpha = np.clip(alpha, 0.0, np.radians(85.0))

            # 1. Translational Forces (LEV-augmented)
            c_l = self.spec.c_l_max * np.sin(2.0 * alpha) * circ_mult
            c_d = self.spec.c_d_0 + self.spec.c_d_max * (1.0 - np.cos(2.0 * alpha))

            q_inf = 0.5 * self.spec.air_density * (speed ** 2) * dS
            dL = q_inf * c_l
            dD = q_inf * c_d
            dF_trans = dL * e_L - dD * e_D

            # 2. Rotational Force (Kramer effect)
            dF_rot_mag = (self.spec.c_rot * self.spec.air_density * 
                          (c ** 2) * state.psi_dot * speed * self.dr)
            dF_rot = np.array([0.0, 0.0, dF_rot_mag])

            # 3. Added-Mass Inertia
            a_norm = -r * state.phi_ddot * np.sin(alpha)
            dF_acc_mag = (np.pi / 4.0) * self.spec.air_density * (c ** 2) * a_norm * self.dr * self.spec.c_added_mass
            dF_acc = np.array([0.0, 0.0, dF_acc_mag])

            dF_elem = dF_trans + dF_rot + dF_acc
            total_F_sp += dF_elem
            total_M_sp += np.cross(r_elem, dF_elem)

            total_lift += dL
            total_drag += dD
            total_rot += abs(dF_rot_mag)
            total_added_mass += abs(dF_acc_mag)

        # Rotate forces and moments from stroke plane frame to body frame
        F_body = R_sp2b @ total_F_sp
        M_body = R_sp2b @ total_M_sp

        telemetry = {
            "lift_n": total_lift,
            "drag_n": total_drag,
            "rotational_n": total_rot,
            "added_mass_n": total_added_mass,
            "net_vertical_n": F_body[2],
            "net_thrust_n": F_body[0],
            "fling_boost_active": fling_active
        }

        return F_body, M_body, telemetry
