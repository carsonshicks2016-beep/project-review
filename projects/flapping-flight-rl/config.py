"""
Physical and biological constants for insect flight simulation.
Benchmark specimen: Manduca sexta (Hawkmoth scale) / Bombus (large bumblebee).
All units are SI (kg, m, s, N, rad).
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class InsectSpec:
    # --- Morphometrics ---
    name: str = "manduca_sexta"
    body_mass: float = 1.6e-3          # 1.6 grams (kg)
    body_length: float = 0.045         # 45 mm (m)
    body_radius: float = 0.007         # 7 mm thorax radius (m)
    
    # Body inertia tensor (approximated as prolate ellipsoid)
    # I_xx (roll), I_yy (pitch), I_zz (yaw)
    I_xx: float = 8.5e-7               # kg * m^2
    I_yy: float = 4.2e-6               # kg * m^2
    I_zz: float = 4.2e-6               # kg * m^2

    # --- Wing Geometry (per wing) ---
    wing_length: float = 0.050         # Span R = 50 mm (m)
    mean_chord: float = 0.018          # Mean chord c_bar = 18 mm (m)
    wing_mass: float = 0.045e-3        # 45 mg per wing (kg)
    wing_area: float = 8.5e-4          # S = 8.5 cm^2 per wing (m^2)
    hinge_offset_x: float = 0.005      # Longitudinal offset from CoM (m)
    hinge_offset_y: float = 0.008      # Lateral offset from midline (m)
    hinge_offset_z: float = 0.004      # Dorsal offset from midline (m)
    blade_elements: int = 10           # Spanwise strips for blade-element integration

    # --- Aerodynamics Coefficients (Dickinson & Sane Model) ---
    air_density: float = 1.205         # rho (kg/m^3) at sea level 20C
    c_l_max: float = 1.95              # LEV-augmented peak translational lift coefficient
    c_d_0: float = 0.15                # Minimum profile drag coefficient
    c_d_max: float = 3.00              # Peak drag coefficient at 90 deg AoA
    c_rot: float = 1.65                # Rotational lift coefficient (Kramer effect)
    c_added_mass: float = 1.00         # Added-mass coefficient (pi/4 * rho * c^2)
    
    # --- Clap-and-Fling (Weis-Fogh) ---
    clap_distance_threshold: float = 0.035   # Inter-wing tip distance for clap detection (35 mm)
    clap_boost_factor: float = 1.35          # Circulation boost multiplier upon fling

    # --- Nominal Baseline Kinematics ---
    nominal_frequency: float = 27.0    # 27 Hz flapping stroke
    stroke_amplitude: float = np.radians(125.0) # Stroke amplitude Phi (~125 degrees)
    stroke_mean_bias: float = np.radians(20.0)  # Mean backward/dorsal stroke sweep
    stroke_plane_angle: float = np.radians(20.0) # Angle of stroke plane relative to body axis

    @property
    def weight(self) -> float:
        return self.body_mass * 9.80665

@dataclass(frozen=True)
class SimConfig:
    sim_dt: float = 1.0 / 2500.0       # 2500 Hz internal physics integration dt
    control_dt: float = 1.0 / 250.0    # 250 Hz agent policy control rate (10 physics steps per action)
    sub_steps: int = 10                # int(control_dt / sim_dt)
