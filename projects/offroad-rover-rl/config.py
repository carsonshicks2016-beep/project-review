"""
Configuration parameters for Off-Road Rover Neuroevolution & Terrain Traversal Simulation.
All units are in SI: meters (m), kilograms (kg), seconds (s), Newtons (N), radians (rad).
"""
import numpy as np

# Simulation Time & Frequency
SIM_DT = 0.01          # 100 Hz internal physics integration
CONTROL_FREQ = 20      # 20 Hz policy decision loop
SUB_STEPS = int((1.0 / CONTROL_FREQ) / SIM_DT)  # 5 sub-steps per policy step
GRAVITY = 9.81         # m/s^2

# Rover Physical Characteristics (Overland 4x4 Buggy)
ROVER_MASS = 800.0     # Total chassis mass (kg)
ROVER_LENGTH = 3.2     # m
ROVER_WIDTH = 1.8      # m
ROVER_HEIGHT = 1.1     # m

# Mass Moments of Inertia (Diagonal approximation around Principal Axes)
# Ixx (Roll), Iyy (Pitch), Izz (Yaw)
INERTIA_XX = (1.0 / 12.0) * ROVER_MASS * (ROVER_WIDTH**2 + ROVER_HEIGHT**2)
INERTIA_YY = (1.0 / 12.0) * ROVER_MASS * (ROVER_LENGTH**2 + ROVER_HEIGHT**2)
INERTIA_ZZ = (1.0 / 12.0) * ROVER_MASS * (ROVER_LENGTH**2 + ROVER_WIDTH**2)
ROVER_INERTIA = np.array([INERTIA_XX, INERTIA_YY, INERTIA_ZZ])

# Center of Mass offset relative to geometric chassis center
COM_OFFSET = np.array([0.0, 0.0, -0.15])  # Slightly lowered for stability

# Wheelbase & Track Geometry
WHEELBASE = 2.2        # Distance between front and rear axles (m)
TRACK_WIDTH = 1.6      # Distance between left and right wheels (m)
HALF_WHEELBASE = WHEELBASE / 2.0
HALF_TRACK = TRACK_WIDTH / 2.0

# 4-Wheel Hardpoints relative to chassis center (x: forward, y: left, z: up)
# Order: 0: Front-Left (FL), 1: Front-Right (FR), 2: Rear-Left (RL), 3: Rear-Right (RR)
WHEEL_ATTACH_POINTS = np.array([
    [HALF_WHEELBASE,  HALF_TRACK, 0.0],   # FL
    [HALF_WHEELBASE, -HALF_TRACK, 0.0],   # FR
    [-HALF_WHEELBASE,  HALF_TRACK, 0.0],  # RL
    [-HALF_WHEELBASE, -HALF_TRACK, 0.0]   # RR
])

# Suspension Kinematics & Tuning
SUSP_REST_LENGTH = 0.55       # Unloaded strut extension (m)
SUSP_MIN_LENGTH = 0.20        # Full bump-stop compression (m)
SUSP_MAX_LENGTH = 0.68        # Full rebound extension (m)
SUSP_SPRING_K = 18500.0       # Spring stiffness (N/m per wheel)
SUSP_DAMPER_C = 2400.0        # Shock absorber damping (N*s/m)
ANTI_ROLL_K = 3500.0          # Anti-roll bar coupling stiffness (N/m)

# Wheel & Tire Dynamics
WHEEL_RADIUS = 0.42           # Tire outer radius (m)
WHEEL_MASS = 28.0             # kg per wheel assembly
WHEEL_INERTIA = 0.5 * WHEEL_MASS * WHEEL_RADIUS**2
TIRE_FRICTION_COEFF = 1.15    # Peak dry rock/gravel friction coefficient (mu)
TIRE_LATERAL_STIFFNESS = 18.0 # Cornering stiffness factor

# Powertrain & Steering Limits
MAX_DRIVE_TORQUE = 2400.0     # Peak combined axle torque (N*m)
MAX_BRAKE_TORQUE = 3200.0     # Peak braking torque (N*m)
MAX_STEER_ANGLE = np.radians(32.0)  # Max front wheel steering (rad, ~0.558 rad)
MAX_STEER_RATE = np.radians(120.0)  # Steer actuator slew rate (rad/s)

# Sensor Array Configuration (16 Raycasts)
RAY_MAX_DIST = 10.0           # Maximum laser range (m)
# Forward fan angles (rad)
FORWARD_FAN_ANGLES = np.radians(np.linspace(-55, 55, 10))
# Lookahead downward pitch angle
FORWARD_FAN_PITCH = np.radians(-18.0)
# Near-bumper obstacle detection angles
NEAR_BUMPER_ANGLES = np.radians([-30, -10, 10, 30])
NEAR_BUMPER_PITCH = np.radians(-42.0)
# Wheel track scan lines
WHEEL_SCAN_OFFSETS = np.array([
    [HALF_WHEELBASE + 0.8,  HALF_TRACK],
    [HALF_WHEELBASE + 0.8, -HALF_TRACK]
])

# Terrain Generation Specs
TERRAIN_LENGTH = 150.0        # Course length along X (m)
TERRAIN_WIDTH = 40.0          # Course width along Y (m)
TERRAIN_RESOLUTION = 0.5      # Heightmap grid resolution (m)
MAX_ELEVATION_GAIN = 14.0     # Steepest hill climb (m)

# Training & Neuroevolution Hyperparameters
POPULATION_SIZE = 48          # Number of genomes per generation
ELITISM_COUNT = 6             # Top performers preserved unconditionally
MUTATION_RATE = 0.12          # Probability of gene mutation
MUTATION_STRENGTH = 0.25      # Standard deviation of Gaussian perturbation
TOURNAMENT_SIZE = 4           # Selection tournament size
MAX_EPISODE_SECONDS = 25.0    # Max time per evaluation trial
MAX_STEPS_PER_TRIAL = int(MAX_EPISODE_SECONDS * CONTROL_FREQ)  # 500 steps
FLIP_ROLL_THRESHOLD = np.radians(68.0)   # Flip detection angle
FLIP_PITCH_THRESHOLD = np.radians(65.0)  # Nose-up / Nose-down flip threshold
