"""
Central configuration for the Supra Drift simulator.

Everything tunable lives here as dataclasses, in SI units (kg, m, s, N, rad).
The physics engine and the rest of the project read from these specs so there
is exactly one place to twist a knob.

This is the foundation slice: vehicle + simulation specs. Sensor / learning
specs are stubbed in so later slices (vision, GA, PPO) have a home, but the
drivable slice only needs CarSpec + SimSpec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# --------------------------------------------------------------------------- #
# Tyre model
# --------------------------------------------------------------------------- #
@dataclass
class Pacejka:
    """Pacejka "Magic Formula" shape coefficients (normalized, peak handled by load).

        F = D * sin(C * atan(B*s - E*(B*s - atan(B*s))))

    where D (peak) = mu * Fz is applied at force-evaluation time, and `s` is the
    slip variable (slip angle in rad for lateral, slip ratio for longitudinal).
    """
    B: float  # stiffness
    C: float  # shape
    E: float  # curvature


# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
@dataclass
class CarSpec:
    name: str = "supra"

    # --- chassis / mass ---
    mass: float = 1540.0          # curb mass (kg)
    wheelbase: float = 2.55       # axle-to-axle (m)
    track_width: float = 1.55     # left-right tyre centre distance (m)
    cg_height: float = 0.46       # centre of gravity height (m)
    front_weight: float = 0.53    # static fraction of weight on the front axle
    yaw_inertia: float = 2400.0   # Iz about the vertical axis (kg m^2)

    # --- tyres ---
    mu: float = 1.15                              # peak friction coefficient
    wheel_radius: float = 0.33                    # effective rolling radius (m)
    wheel_inertia: float = 1.2                    # per-wheel spin inertia (kg m^2)
    load_sensitivity: float = 5.0e-5             # mu falloff per N above nominal Fz
    pacejka_lat: Pacejka = field(default_factory=lambda: Pacejka(B=9.5, C=1.45, E=0.97))
    pacejka_long: Pacejka = field(default_factory=lambda: Pacejka(B=14.0, C=1.65, E=0.90))
    rolling_resistance: float = 0.013            # Crr (force = Crr * Fz, opposes motion)
    relaxation_length: float = 0.55              # lateral tyre force build-up distance (m)
    relaxation_length_long: float = 0.25         # longitudinal (short: stabilises the
                                                 # low-speed slip-ratio singularity
                                                 # without bogging launches)

    # --- handling balance ---
    # fraction of lateral load transfer carried by the FRONT axle (roll-stiffness
    # distribution). >0.5 => more front transfer => understeer bias; <0.5 => oversteer.
    roll_front_frac: float = 0.52
    # steering
    steer_angle_max_deg: float = 32.0            # road-wheel angle at full lock
    steer_rate_deg_s: float = 360.0              # max steering rate (deg/s)
    # rear steer / HICAS-style systems (0 disables)
    rear_steer_max_deg: float = 0.0              # rear-wheel steering angle clamp
    rear_steer_low_speed_gain: float = -0.10     # opposite-phase blend at low speed
    rear_steer_high_speed_gain: float = 0.08     # same-phase blend at higher speed
    rear_steer_transition_mps: float = 14.0      # speed where high-speed phase fades in
    # off-track surface grip multiplier (grass/dirt vs tarmac)
    offtrack_grip: float = 0.7

    # --- aero ---
    air_density: float = 1.225
    drag_area: float = 0.62       # Cd * frontal area (m^2), longitudinal drag
    side_drag_area: float = 1.9   # effective side area for lateral drag (m^2)
    downforce_ClA: float = 0.95   # Cl * area; downforce = 0.5*rho*ClA*v^2 (N)
    aero_balance: float = 0.45    # fraction of downforce on the front axle

    # --- powertrain ---
    engine_inertia: float = 0.35                 # flywheel + rotating mass (kg m^2)
    idle_rpm: float = 850.0
    redline_rpm: float = 6800.0
    cutoff_rpm: float = 7000.0
    engine_friction: float = 0.04                # internal drag torque per rad/s
    idle_gain: float = 6.0                        # idle governor torque per rad/s below idle
    idle_torque_max: float = 35.0                # cap on idle governor torque (Nm)
    # torque curve: (rpm, torque Nm) at full throttle & full boost
    torque_curve: List[Tuple[float, float]] = field(default_factory=lambda: [
        (850, 240), (1500, 360), (2200, 470), (3000, 510),
        (4000, 520), (5000, 500), (6000, 450), (6800, 400), (7000, 0),
    ])
    # Getrag-style 6-speed + final drive
    gear_ratios: List[float] = field(default_factory=lambda: [3.83, 2.36, 1.69, 1.31, 1.00, 0.79])
    final_drive: float = 3.15
    drivetrain_efficiency: float = 0.90
    clutch_capacity: float = 900.0               # max torque the clutch can hold (Nm)
    clutch_slip_ref: float = 28.0                # rad/s scale for clutch engagement
                                                 # (softer = stable near lock-up)

    # --- twin-turbo (sequential) ---
    boost_floor: float = 0.55      # fraction of torque available off-boost
    spool_up_tau: float = 0.35     # spool lag time constant when building boost (s)
    spool_down_tau: float = 0.12   # bleed-off time constant when off throttle (s)

    # --- limited-slip differential (REAR axle) ---
    lsd_coef: float = 120.0        # locking torque per rad/s of wheel speed delta

    # --- drivetrain layout ---
    drive_layout: str = "rwd"          # "rwd" | "fwd" | "awd"
    center_split_front: float = 0.0    # AWD: fraction of drive torque to FRONT axle
                                       #   rwd=0.0, fwd=1.0, awd e.g. 0.5 (LR4) / 0.35 (R34)
    center_diff: str = "open"          # "open" | "lsd" | "locked" | "attesa" (AWD only)
    center_lsd_coef: float = 150.0     # locking torque per rad/s of front-rear shaft delta
    front_lsd_coef: float = 0.0        # front-axle diff locking (0 = open)
    # ATTESA-like active transfer case: mostly rear-drive, feed front axle under
    # launch/rear-slip/g-load demand. No effect unless center_diff == "attesa".
    attesa_front_min: float = 0.0
    attesa_front_max: float = 0.50
    attesa_launch_split: float = 0.0
    attesa_slip_gain: float = 0.0
    attesa_g_gain: float = 0.0
    attesa_response_tau: float = 0.08

    # --- coupling (engine -> gearbox) ---
    coupling: str = "clutch"           # "clutch" | "torque_converter"
    # torque-converter params (used only when coupling == "torque_converter")
    tc_capacity: float = 3.5e-4        # pump capacity: T_pump = tc_capacity*w_imp^2*(1-sr^2)
    tc_mult_max: float = 2.2           # torque multiplication at stall (speed ratio 0)
    tc_coupling_sr: float = 0.88       # speed ratio where torque ratio reaches 1.0
    tc_lockup_sr: float = 0.92         # speed ratio above which the lockup clutch blends in

    # --- body roll exaggeration (visual lean in carart; 1.0 = sports car) ---
    body_roll_gain: float = 1.0

    # --- brakes ---
    max_brake_torque: float = 2400.0   # total at full pedal (Nm), split by bias
    brake_bias: float = 0.62           # fraction to the front axle
    handbrake_torque: float = 3200.0   # rear-axle lock torque at full handbrake

    # --- suspension settling ---
    suspension_tau: float = 0.14   # first-order lag on vertical load transfer (s)

    # --- airborne / landing (PHYSICS_3D_PLAN Stage 3) ---
    landing_tau: float = 0.12      # s — suspension absorbs the touchdown impulse
    air_pitch_rate: float = 1.2    # rad/s — nose eases toward the flight arc
    max_landing_load: float = 4.0  # impact load clamp, x total static weight

    # ----- derived geometry helpers ----- #
    @property
    def a(self) -> float:
        """Distance from CG to the front axle (m). Front-heavy -> smaller a."""
        return (1.0 - self.front_weight) * self.wheelbase

    @property
    def b(self) -> float:
        """Distance from CG to the rear axle (m)."""
        return self.front_weight * self.wheelbase

    @property
    def half_track(self) -> float:
        return self.track_width / 2.0


# --------------------------------------------------------------------------- #
# Vehicle presets
# --------------------------------------------------------------------------- #
def supra() -> CarSpec:
    """Toyota Supra (A90-ish) — the default, balanced grip + drift platform."""
    return CarSpec(name="supra")


def rx7() -> CarSpec:
    """Mazda RX-7 (FD) — light, tail-happy, peaky rotary-ish torque."""
    return CarSpec(
        name="rx7",
        mass=1310.0,
        wheelbase=2.43,
        track_width=1.49,
        cg_height=0.43,
        front_weight=0.505,
        yaw_inertia=1950.0,
        mu=1.12,
        torque_curve=[
            (850, 150), (2000, 230), (3500, 300), (4500, 360),
            (5500, 380), (6500, 360), (7500, 320), (8200, 0),
        ],
        redline_rpm=7800.0,
        cutoff_rpm=8200.0,
        gear_ratios=[3.48, 2.02, 1.39, 1.00, 0.76, 0.62],
        final_drive=4.10,
        boost_floor=0.45,
        engine_inertia=0.18,        # low-inertia rotary flywheel
        spool_up_tau=0.25,          # responsive twin-turbo sequential setup
        spool_down_tau=0.10,
    )


def skyline() -> CarSpec:
    """Nissan Skyline GT-R (R34) — RB26DETT, ATTESA E-TS Pro, Super-HICAS."""
    return CarSpec(
        name="skyline",
        mass=1560.0,
        wheelbase=2.665,
        track_width=1.48,
        cg_height=0.47,
        front_weight=0.56,
        yaw_inertia=2550.0,
        mu=1.10,
        wheel_radius=0.327,          # 245/40R18
        wheel_inertia=1.35,
        roll_front_frac=0.56,        # nose-heavy GT-R: stable, mild understeer
        rear_steer_max_deg=0.9,      # Super-HICAS is subtle, not forklift steering
        rear_steer_low_speed_gain=-0.12,
        rear_steer_high_speed_gain=0.10,
        rear_steer_transition_mps=13.0,
        drag_area=0.72,
        downforce_ClA=0.28,          # modest factory aero/diffuser effect
        aero_balance=0.42,
        torque_curve=[
            # Healthy factory RB26DETT. Published numbers were conservative;
            # this keeps the real broad midrange without the old tuned 540 Nm
            # plateau. Boost lag is applied separately below.
            (850, 160), (1600, 255), (2400, 365), (3200, 440),
            (4400, 465), (5600, 445), (6800, 350), (7600, 295), (8200, 0),
        ],
        redline_rpm=8000.0,
        cutoff_rpm=8200.0,
        gear_ratios=[3.83, 2.36, 1.69, 1.31, 1.00, 0.79],
        final_drive=3.545,
        drivetrain_efficiency=0.88,  # transfer case + front driveline losses
        clutch_capacity=1200.0,
        boost_floor=0.64,            # enough off-boost torque for the sim clutch
        engine_inertia=0.34,
        spool_up_tau=0.30,
        spool_down_tau=0.11,
        drive_layout="awd",
        center_split_front=0.0,      # ATTESA baseline: rear-drive until demanded
        center_diff="attesa",
        attesa_front_min=0.0,
        attesa_front_max=0.50,       # transfer case can approach 50:50
        attesa_launch_split=0.20,    # preload front torque on hard launches
        attesa_slip_gain=0.70,       # rear wheelspin quickly asks for front assist
        attesa_g_gain=0.12,          # G-sensor style stabilising assist
        attesa_response_tau=0.09,
        lsd_coef=155.0,              # mechanical/active rear LSD behavior
        front_lsd_coef=0.0,          # front diff effectively open
    )


def lr4() -> CarSpec:
    """2012 Land Rover LR4 HSE — tall, heavy, NA 5.0 V8, full-time AWD, soft air
    suspension. The understeery anti-sports-car: big roll, lazy yaw, planted."""
    return CarSpec(
        name="lr4",
        # chassis / mass (tall + heavy SUV)
        mass=2650.0,
        wheelbase=2.885,
        track_width=1.605,
        cg_height=0.86,            # ~2x a sports car -> huge load transfer + roll
        front_weight=0.51,
        yaw_inertia=4800.0,        # long + heavy -> lazy rotation
        # tyres (tall all-season, lower grip)
        mu=1.00,
        wheel_radius=0.38,         # 255/55 R19
        wheel_inertia=2.4,         # big heavy wheels
        load_sensitivity=6.0e-5,
        rolling_resistance=0.016,
        # handling balance: understeer bias + slow soft responses
        roll_front_frac=0.58,      # >0.5 -> understeer
        steer_angle_max_deg=34.0,
        steer_rate_deg_s=220.0,    # slow rack
        offtrack_grip=0.85,        # it's actually good off-tarmac
        # aero: a brick with a roof rack, no downforce
        drag_area=1.35,
        side_drag_area=3.2,
        downforce_ClA=0.05,
        aero_balance=0.5,
        # powertrain: NA 5.0 V8 (375 hp / ~508 Nm @ 3500), 6500 redline
        engine_inertia=0.42,
        idle_rpm=650.0,
        redline_rpm=6500.0,
        cutoff_rpm=6700.0,
        torque_curve=[
            (650, 300), (1500, 400), (2500, 470), (3500, 508),
            (4500, 495), (5500, 460), (6500, 400), (6700, 0),
        ],
        # ZF 6HP28 6-speed automatic + final drive
        gear_ratios=[4.17, 2.34, 1.52, 1.14, 0.87, 0.69],
        final_drive=3.54,
        drivetrain_efficiency=0.85,   # AWD + auto losses
        # NA: no boost dependence (torque curve is the whole story)
        boost_floor=1.0,
        # FULL driveline: torque converter + AWD locking centre + rear diff
        coupling="torque_converter",
        tc_capacity=8.0e-3, tc_mult_max=2.2, tc_coupling_sr=0.88, tc_lockup_sr=0.92,
        drive_layout="awd",
        center_split_front=0.5,       # symmetric full-time 4WD
        center_diff="locked",         # locking centre diff
        center_lsd_coef=200.0,
        lsd_coef=140.0,               # rear diff
        front_lsd_coef=60.0,          # mild front
        # brakes: big but soft pedal, lots of dive
        max_brake_torque=3200.0,
        brake_bias=0.60,
        handbrake_torque=2600.0,
        # soft air suspension -> slow load settling = visible roll/dive
        suspension_tau=0.26,
        body_roll_gain=1.8,           # tall SUV leans hard (carart visual)
    )


def f150() -> CarSpec:
    """2025 Ford F-150 XLT — 5.0L Coyote V8 (NA, 400 hp / 556 Nm @ 4250), 10-speed
    automatic, rear-wheel drive (a real 4x2, or a 4x4 left in 2H on pavement). Long,
    heavy, front-biased pickup: lazy to rotate, understeery, lots of squat/dive,
    big lazy NA-V8 torque. Flip drive_layout->'awd' + center_split_front=0.5 for 4x4."""
    return CarSpec(
        name="f150",
        # chassis / mass (SuperCrew 145" wheelbase, heavy, front-heavy empty bed)
        mass=2200.0,
        wheelbase=3.68,
        track_width=1.72,
        cg_height=0.75,            # high pickup CG
        front_weight=0.58,         # nose-heavy with an empty bed
        yaw_inertia=5200.0,        # very long + heavy -> reluctant to rotate
        # tyres (tall all-season truck tyres, modest grip)
        mu=0.95,
        wheel_radius=0.39,         # ~265/70R17
        wheel_inertia=2.6,         # big heavy wheels
        load_sensitivity=6.0e-5,
        rolling_resistance=0.016,
        # handling: understeer bias, slow soft responses
        roll_front_frac=0.58,
        steer_angle_max_deg=36.0,
        steer_rate_deg_s=240.0,    # slow truck rack
        offtrack_grip=0.80,
        suspension_tau=0.20,       # softish leaf-sprung rear -> pitch/squat
        body_roll_gain=1.6,        # tall body leans (carart visual; you do visuals)
        # aero: a brick, no downforce
        drag_area=1.45,
        side_drag_area=3.6,
        downforce_ClA=0.05,
        aero_balance=0.5,
        # engine: 5.0 Coyote V8 NA — 400 hp @ 6000, 410 lb-ft (556 Nm) @ 4250
        engine_inertia=0.40,
        idle_rpm=650.0,
        redline_rpm=6500.0,
        cutoff_rpm=6700.0,
        torque_curve=[
            (650, 340), (1500, 440), (2500, 500), (3500, 545),
            (4250, 556), (5000, 540), (6000, 475), (6500, 430), (6700, 0),
        ],
        # Ford 10R80 10-speed automatic + final drive
        gear_ratios=[4.69, 2.98, 2.14, 1.76, 1.52, 1.27, 1.00, 0.85, 0.68, 0.63],
        final_drive=3.55,
        drivetrain_efficiency=0.86,
        # NA: no boost dependence
        boost_floor=1.0,
        # automatic torque converter, rear-wheel drive
        coupling="torque_converter",
        tc_capacity=8.0e-3, tc_mult_max=2.0, tc_coupling_sr=0.88, tc_lockup_sr=0.90,
        drive_layout="rwd",
        lsd_coef=80.0,             # mild limited-slip / available e-locker rear
        # brakes: big truck brakes, soft pedal, lots of dive
        max_brake_torque=3400.0,
        brake_bias=0.60,
        handbrake_torque=2400.0,
    )


def evader() -> CarSpec:
    """Light, high-grip drift weapon for the heist prototype.

    This keeps the Supra physics model intact while making the evader match the
    baseline: low mass, high lateral authority, fast steering, and a serious
    handbrake for 90-degree city pivots.
    """
    return CarSpec(
        name="evader",
        mass=1180.0,
        wheelbase=2.36,
        track_width=1.50,
        cg_height=0.40,
        front_weight=0.49,
        yaw_inertia=1580.0,
        mu=1.34,
        wheel_radius=0.315,
        wheel_inertia=0.85,
        load_sensitivity=3.8e-5,
        roll_front_frac=0.47,
        steer_angle_max_deg=44.0,
        steer_rate_deg_s=680.0,
        offtrack_grip=0.60,
        drag_area=0.56,
        side_drag_area=1.55,
        downforce_ClA=1.15,
        aero_balance=0.42,
        engine_inertia=0.22,
        idle_rpm=950.0,
        redline_rpm=7900.0,
        cutoff_rpm=8200.0,
        torque_curve=[
            (950, 185), (1800, 280), (2800, 390), (4000, 455),
            (5600, 470), (7000, 415), (7900, 330), (8200, 0),
        ],
        gear_ratios=[3.77, 2.22, 1.62, 1.27, 1.00, 0.82],
        final_drive=4.10,
        drivetrain_efficiency=0.91,
        clutch_capacity=980.0,
        boost_floor=0.58,
        spool_up_tau=0.22,
        spool_down_tau=0.08,
        drive_layout="rwd",
        lsd_coef=170.0,
        max_brake_torque=2550.0,
        brake_bias=0.58,
        handbrake_torque=4700.0,
        suspension_tau=0.10,
        body_roll_gain=0.85,
    )


def pursuer() -> CarSpec:
    """Heavy pursuit interceptor.

    The car has speed and torque, but its mass, yaw inertia, slower rack, and
    lower tyre peak make it punish sharp imitation of the evader's drift turns.
    """
    return CarSpec(
        name="pursuer",
        mass=2140.0,
        wheelbase=2.94,
        track_width=1.64,
        cg_height=0.59,
        front_weight=0.58,
        yaw_inertia=4050.0,
        mu=0.92,
        wheel_radius=0.345,
        wheel_inertia=1.85,
        load_sensitivity=5.8e-5,
        rolling_resistance=0.015,
        roll_front_frac=0.63,
        steer_angle_max_deg=31.0,
        steer_rate_deg_s=250.0,
        offtrack_grip=0.50,
        drag_area=0.86,
        side_drag_area=2.45,
        downforce_ClA=0.22,
        aero_balance=0.50,
        engine_inertia=0.43,
        idle_rpm=700.0,
        redline_rpm=6600.0,
        cutoff_rpm=6900.0,
        torque_curve=[
            (700, 260), (1500, 390), (2600, 515), (3900, 560),
            (5200, 535), (6200, 455), (6900, 0),
        ],
        gear_ratios=[4.70, 3.15, 2.10, 1.67, 1.29, 1.00, 0.84, 0.67],
        final_drive=3.23,
        drivetrain_efficiency=0.86,
        boost_floor=1.0,
        coupling="torque_converter",
        tc_capacity=8.0e-3,
        tc_mult_max=2.0,
        tc_coupling_sr=0.88,
        tc_lockup_sr=0.91,
        drive_layout="awd",
        center_split_front=0.46,
        center_diff="locked",
        center_lsd_coef=190.0,
        front_lsd_coef=35.0,
        lsd_coef=110.0,
        max_brake_torque=3350.0,
        brake_bias=0.63,
        handbrake_torque=1500.0,
        suspension_tau=0.20,
        body_roll_gain=1.45,
    )


PRESETS = {
    "supra": supra,
    "rx7": rx7,
    "skyline": skyline,
    "lr4": lr4,
    "f150": f150,
    "evader": evader,
    "pursuer": pursuer,
}


def get_car(name: str = "supra") -> CarSpec:
    name = name.lower()
    if name not in PRESETS:
        raise ValueError(f"Unknown car '{name}'. Choose from {list(PRESETS)}.")
    return PRESETS[name]()


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimSpec:
    dt: float = 1.0 / 120.0          # physics timestep (s)
    drivetrain_substeps: int = 6     # sub-stepping for stiff driveline / tyre spin
    velocity_eps: float = 0.5        # m/s floor for slip denominators
    relax_speed_floor: float = 1.0   # m/s floor for relaxation-length time constant
    fps: int = 120                   # render / display rate (ProMotion 120 Hz cap)


# --------------------------------------------------------------------------- #
# Sensors (placeholder for the vision slice)
# --------------------------------------------------------------------------- #
@dataclass
class SensorSpec:
    n_beams: int = 9
    beam_spread_deg: float = 100.0           # +/- coverage
    beam_range: float = 70.0                 # m
    lookahead_distances: Tuple[float, ...] = (8.0, 16.0, 28.0, 44.0, 64.0, 90.0)
    n_proprio: int = 25

    # --- hill/air block (PHYSICS_3D_PLAN Stage 5; obs 40 -> 58) ---
    hill_block: bool = True        # debug escape hatch only — training assumes True
    grade_ref: float = 0.20        # slope normalization (~±11°)
    bank_ref: float = 0.10
    pitch_ref: float = 0.30        # body pitch (rad) — ballistic range in the air
    vz_ref: float = 12.0           # vertical speed (m/s)
    height_ref: float = 3.0        # height above road (m) — landing anticipation
    vcurv_ref: float = 0.02        # crest/dip preview (1/m)


# --------------------------------------------------------------------------- #
# Genetic algorithm (slice 3)
# --------------------------------------------------------------------------- #
@dataclass
class EvoSpec:
    pop_size: int = 60
    hidden: Tuple[int, ...] = (32, 24)        # MLP hidden layers
    n_actions: int = 3                         # steer, throttle, brake (auto gearbox)
    elite_frac: float = 0.15
    tournament_k: int = 3
    mutation_rate: float = 0.18
    mutation_sigma: float = 0.30
    init_weight_scale: float = 0.35            # small so gen-0 drives sensibly

    # episode / fitness
    episode_seconds: float = 45.0
    control_hz: int = 30                       # brain decision rate
    offtrack_timeout: float = 1.2              # s off-track before a crash
    stall_timeout: float = 2.5                 # s under crawl speed before done
    start_grace: float = 1.0                   # s grace at launch
    progress_timeout: float = 4.0              # s without progress before done

    # sensible gen-0 output biases (untrained cars behave reasonably)
    bias_throttle: float = 0.73
    bias_brake: float = 0.18
    bias_clutch: float = 0.88


# --------------------------------------------------------------------------- #
# PPO (slice 4) — hybrid-ready (mode-conditioned race / drift)
# --------------------------------------------------------------------------- #
@dataclass
class PPOSpec:
    # network
    hidden: Tuple[int, ...] = (128, 128)

    # optimisation
    lr: float = 3.0e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 8
    ent_coef: float = 0.003            # modest: enough exploration, lets std sharpen
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5

    # schedules (anneal over a run) — explore early, sharpen + refine late. Off by
    # default (preserves behaviour); turn on for long runs with --anneal.
    anneal: bool = False
    lr_floor_frac: float = 0.2         # lr decays (cosine) to this fraction by run end
    ent_floor_frac: float = 0.15       # ent_coef decays to this fraction (less noise late)

    # rollout / vectorisation
    n_envs: int = 8
    rollout: int = 256                # steps per env per update (n_envs*rollout total)
    n_workers: int = 1                # >1 -> run the vec env across this many SUBPROCESSES
                                      # (parallel rollouts; 1 = the in-process SyncVecEnv)

    # episode
    control_hz: int = 30
    episode_seconds: float = 60.0

    # curriculum
    start_difficulty: float = 0.15
    promote_at: float = 0.72          # master a level (rolling lap-completion) before
                                      # bumping difficulty — avoids racing to diff 1.0
                                      # while still mediocre on the hard tracks
    difficulty_step: float = 0.08
    max_difficulty: float = 1.0
    drift_promote_at: float = 0.30    # drift: promote on time-spent-drifting, not laps
                                      # (it plateaued ~0.35 at diff 0.15, so 0.45 never
                                      # advanced it — 0.30 lets the curriculum progress)
    drift_promote_lap: float = 0.45   # ...but ALSO require it to actually get around
                                      # the loop before advancing, so the curriculum
                                      # can't race ahead while the policy just slides
                                      # and spins out (the old "drifts but doesn't
                                      # complete" trap). Rolling lap-completion gate.
    track_pool: int = 16              # distinct tracks per env per level (variety
                                      # vs memorisation — too few overfits)
    random_start: bool = True         # exploring starts: reset the car at a RANDOM
                                      # point along the track, not always the start
                                      # line. Critical for specialists (one fixed
                                      # track) so it practises the WHOLE circuit from
                                      # iter 1 instead of only the opening section it
                                      # can reach before spinning out.
    specialist_patience: int = 500    # specialists plateau (no curriculum), so
                                      # auto-stop when the best score hasn't improved
                                      # in this many iters — keep the peak, skip the
                                      # over-training collapse. Generous, so it never
                                      # bails during the early "learn-to-complete
                                      # then learn-to-drift" phase. 0 disables.


@dataclass
class DriftReward:
    scale: float = 0.015              # overall scale (drift episodes are long, so
                                      # keep per-step small for the value function)
    progress: float = 30.0            # per lap-FRACTION (on-track): make it get
                                      # AROUND the loop, not just slide-then-spin.
                                      # Smaller than race's 60 so the drift core
                                      # still dominates (drift-WHILE-progressing).
    drive: float = 0.05               # SMALL speed bootstrap (on-track only) — just
                                      # enough to get it moving, so clean driving earns
                                      # little and the on-track drift reward is the
                                      # clear prize (a bigger bootstrap makes safe
                                      # clean driving competitive and it won't drift).
    angle_speed: float = 3.5          # core: speed * |sin(slip)| * this
    entry_lo_deg: float = 2.3         # soft-entry smoothstep start
    entry_hi_deg: float = 11.5        # soft-entry full by here
    peak_deg: float = 55.0            # reward grows to ~here (big controlled drift)
    spin_deg: float = 92.0            # over-rotation: reward -> 0 beyond (it's a spin)
    # --- anti-spin shaping (the core "slides but spins out" fix) ---
    # The old falloff only watched the STATIC slip angle, so it couldn't tell a
    # held 50 deg drift from a 50 deg snapshot mid-spin until the angle already
    # blew up — too late to stop the policy rotating out of control. A controlled
    # drift holds a steady angle; a spin is runaway YAW RATE. These gate the
    # reward on yaw rate and pay a bonus for holding the angle steady through a
    # corner. Conservative defaults (only bite on genuine spins) — tune + revalidate.
    spin_rate_deg: float = 160.0      # yaw rate (deg/s) above which it's losing control
    spin_rate_span: float = 140.0     # reward fully gated by spin_rate_deg + this
    hold_bonus: float = 1.0           # bonus for a STEADY controlled angle in-band
    hold_rate_deg: float = 35.0       # |d(slip)/dt| (deg/s) under this = "held steady"
    # --- slide RECOVERY (catch it, don't die) ---
    # A spin only fails if you let it run away. Reward bringing the yaw rate back
    # DOWN while over-rotating (the countersteer "save"), and give a wide slide a
    # little more leash before terminating, so the policy can learn to recover a
    # slide onto the road instead of treating every big angle as commit-or-crash.
    recover_bonus: float = 2.5        # reward for actively reducing yaw while over-rotating
    recover_yaw_deg: float = 70.0     # only counts as a "save" above this yaw rate
    offtrack_grace: float = 0.9       # s off-track before terminating (was 0.6 — a bit
                                      # more room to bring a wide slide back on)
    spin_grace: float = 1.2           # s near-stationary before calling it a spin-out
    speed_gate: float = 12.5          # m/s for full speed scaling (~28 mph)
    initiation: float = 2.0           # rear wheelspin at speed (before the slide)
    donut_speed: float = 4.0          # below this + high angle = static donut
    donut_penalty: float = 3.0
    transition_bonus: float = 4.0     # direction change within the chain window
    chain_window: float = 1.8         # s to chain a transition
    sustain_rate: float = 0.5         # sustain multiplier growth per s drifting
    sustain_max: float = 3.0
    gutter_bonus: float = 1.5         # clipping the inner edge while sliding
    drift_min_deg: float = 6.0        # slip angle that counts as "drifting"
    offtrack: float = 0.1             # MILD: the real disincentive for going off is
                                      # the reward GATING (no reward on grass) + the
                                      # episode ending — a harsh penalty just scares
                                      # it out of drifting at all.
    crash: float = 2.0                # terminal penalty (mild, same reason)


@dataclass
class RaceReward:
    progress: float = 60.0            # per lap-FRACTION (length-independent: lap ~ 60)
    align: float = 0.06               # dense: cos(heading err) * speed, per substep
    speed: float = 0.005              # per (speed / 80 m/s), per substep
    offtrack: float = 0.15            # penalty per substep while off-track
    smooth: float = 0.002             # penalty on action change^2
    crash: float = 3.0                # terminal penalty on crash
    lap_bonus: float = 5.0            # bonus for completing a lap


@dataclass
class HybridReward:
    """Race + drift FUSION: a race backbone (complete the lap fast, on track) plus
    a STYLE bonus that fires only IN CORNERS — so it grips the straights for speed
    and drifts the turns for flair. `style` is the master dial: bigger = driftier
    (and a bit slower); too big and it reverts to a pure drifter, too small and it
    just races. Same action space + obs as drift (mode one-hot [1,1]), so a hybrid
    can WARM-START from a drift policy."""
    # --- race backbone (identical knobs to RaceReward) ---
    progress: float = 60.0
    align: float = 0.06
    speed: float = 0.008              # a touch higher than race: reward carrying speed
    offtrack: float = 0.15
    smooth: float = 0.002
    crash: float = 3.0
    lap_bonus: float = 5.0
    # --- drift STYLE bonus (corner-gated, controlled) ---
    style: float = 0.03              # MASTER dial for how drifty it is (tune first)
    style_speed_gate: float = 12.5   # m/s for full style scaling
    entry_lo_deg: float = 6.0        # soft-entry: only real slides earn style
    entry_hi_deg: float = 16.0
    peak_deg: float = 50.0           # controlled-drift peak
    spin_deg: float = 90.0           # over-rotation -> style fades to 0
    spin_rate_deg: float = 160.0     # anti-spin yaw-rate gate (controlled, not a spin)
    spin_rate_span: float = 140.0
    corner_curv: float = 0.012       # |curvature| where a "corner" begins (gate on)
    corner_span: float = 0.012       # gate fully on by corner_curv + this
    drift_min_deg: float = 6.0       # slip that counts as 'drifting' for the metric
