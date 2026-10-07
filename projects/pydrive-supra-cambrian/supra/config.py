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
    durability: float = 1.0       # Chassis armor multiplier (higher = tougher)

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
    drivetrain_version: str = "generic-v1"
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
    is_ttr_hybrid: bool = False        # Through-The-Road hybrid: ICE rear, MGU front
    mgu_gear_ratio: float = 5.5        # Fixed reduction gear for the front MGU
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

    # --- certified body footprint (collision + track-limit OBB) ---
    # 0 = derive from the legacy formula (wheelbase+1.7 x track+0.36).
    # Set explicitly when the real car's overall dimensions are published.
    body_length: float = 0.0
    body_width: float = 0.0

    # --- active aero (LMP1 DRS-style; inert unless aero_active) ---
    aero_active: bool = False          # straightline low-drag mode available
    aero_lowdrag_factor: float = 1.0   # drag_area multiplier in low-drag mode
    aero_lowlift_factor: float = 1.0   # downforce_ClA multiplier in low-drag mode

    # --- hybrid energy ledger (inert unless hybrid_mgu_power_w > 0) ---
    # The torque_curve is the COMBINED (ICE + MGU) full-deployment curve.
    # The ledger caps the MGU share by battery state: deployment drains the
    # battery, braking regenerates it, and an empty battery derates the car
    # to ICE-only power. Hybrid cars expose SOC + MGU power to the obs via
    # SensorSpec.hybrid_block (fable-v2); non-hybrid layouts never see it.
    hybrid_mgu_power_w: float = 0.0    # MGU-K peak electric power (W)
    hybrid_mgu_peak_torque_nm: float = 0.0  # MGU-K peak MOTOR-shaft torque.
                                       # An electric motor is constant-TORQUE
                                       # below its base speed, constant-POWER
                                       # above. Without this cap, P/omega -> inf
                                       # as speed -> 0 (58 kNm at standstill),
                                       # exploding front wheelspin. 0 = uncapped
                                       # (non-hybrid cars never reach this).
    hybrid_mguh_power_w: float = 0.0   # MGU-H exhaust harvest while on throttle (W)
    hybrid_ice_power_frac: float = 1.0 # ICE share of the combined curve
    hybrid_battery_kj: float = 0.0     # usable battery energy (kJ)
    hybrid_regen_eff: float = 0.0      # brake energy -> battery efficiency
    mgu_tc_lat_g: float = 1.5          # TTR traction control: front-axle MGU
                                       # deployment fades to zero by this
                                       # lateral load (the fronts are steering)

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
    return CarSpec(name="supra", durability=1.2)


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
        durability=0.8,              # fragile chassis, complex turbo system
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


def mazda787b() -> CarSpec:
    """Mazda 787B — Group C legend, 4-rotor NA R26B, extremely lightweight."""
    return CarSpec(
        name="mazda787b",
        # --- chassis / mass ---
        mass=830.0,                  # incredibly lightweight
        wheelbase=2.66,
        track_width=1.53,
        cg_height=0.32,              # sits millimeters off the ground
        front_weight=0.45,           # mid-engine bias
        yaw_inertia=1100.0,
        durability=0.4,              # extremely fragile carbon-kevlar shell
        
        # --- tyres ---
        mu=1.80,                     # racing slicks
        wheel_radius=0.33,           # wide low-profile racing tires
        wheel_inertia=0.9,           # light magnesium wheels
        load_sensitivity=2.0e-5,     # slicks handle load well
        pacejka_lat=Pacejka(B=12.0, C=1.6, E=0.95),  # sharp, stiff racing slip curve
        pacejka_long=Pacejka(B=15.0, C=1.7, E=0.95),
        rolling_resistance=0.012,
        relaxation_length=0.20,      # stiff racing sidewalls (immediate response)
        relaxation_length_long=0.15,

        # --- handling balance ---
        roll_front_frac=0.45,        # biased for rear grip
        steer_angle_max_deg=25.0,    # limited steering lock (race car)
        steer_rate_deg_s=500.0,      # ultra-fast rack
        offtrack_grip=0.5,           # slicks on grass = ice
        
        # --- aero ---
        drag_area=0.85,              # high drag due to massive wings
        side_drag_area=2.5,          # slab-sided prototype
        downforce_ClA=3.5,           # massive downforce
        aero_balance=0.40,           # downforce biased to the rear
        
        # --- powertrain (R26B 4-Rotor NA) ---
        drivetrain_version="mazda787b-5spd-ring-v1",
        engine_inertia=0.10,         # basically no flywheel, revs instantly
        idle_rpm=2000.0,             # race idle
        redline_rpm=9000.0,
        cutoff_rpm=9000.0,
        engine_friction=0.08,
        idle_gain=10.0,
        # Official R26B anchors: 62 kg-m @ 6500 rpm and 700 PS @ 9000 rpm.
        # The 9000-rpm point is ~546 Nm; the limiter prevents extrapolation.
        torque_curve=[
            (2000, 200), (3500, 320), (5000, 480), (6500, 608),
            (7500, 595), (8500, 565), (9000, 546),
        ],
        # Mazda-Porsche five-speed architecture. These Ring ratios are selected
        # by tools/optimize_787b_gearing.py and frozen for this drivetrain version.
        gear_ratios=[3.27451, 2.28028, 1.57271, 1.16654, 0.96318],
        final_drive=3.45543,
        drivetrain_efficiency=0.95,  # straight-cut racing transaxle
        clutch_capacity=1500.0,      # holds massive power
        clutch_slip_ref=15.0,        # carbon racing clutch, grabs violently
        boost_floor=1.0,             # NA
        
        # --- drivetrain layout ---
        drive_layout="rwd",
        lsd_coef=300.0,              # very tight differential
        
        # --- suspension / visual ---
        body_roll_gain=0.15,         # ultra-stiff, almost zero visual body roll
        suspension_tau=0.05,         # immediate settling
        
        # --- brakes ---
        # 8000 Nm ≈ 2.9 g of pure brake authority (8000/0.33/830) — Group C
        # carbon-brake territory. 6000 Nm gave only 2.2 g, below the era's
        # documented ~3 g braking performance.
        max_brake_torque=8000.0,     # carbon-carbon stopping power
        brake_bias=0.55,
        handbrake_torque=0.0,        # no handbrake on a Group C car
    )


def porsche_956() -> CarSpec:
    """Porsche 956 (1983 qualifying trim) — the car Stefan Bellof drove to the
    6:11.13 Nordschleife pole. Type 935 2.65 L twin-turbo flat-6, full
    ground-effect underbody (venturi tunnels — far more downforce than the
    flat-bottomed 1991 787B). Modelled so a faithful racing-line lap brackets
    Bellof's 371.13 s, i.e. the human benchmark sits right at the car's limit.

    Defensible anchors (well-documented): ~820 kg (Group C 800 kg minimum),
    ~630 hp on qualifying boost, 5-speed, ground-effect downforce ~2.5x weight
    at speed. The one calibrated figure is downforce_ClA (no public exact value
    for a 1983 956); it is tuned so the centerline envelope lands ~415 s and the
    racing line ~371 s. Grip/aero here are HIGHER than the 787B on purpose —
    this is a genuinely faster car, which is why it (not the 787B) set 371 s.
    """
    return CarSpec(
        name="porsche_956",
        # --- chassis / mass ---
        mass=820.0,                  # Group C minimum-ish, qualifying trim
        wheelbase=2.65,              # published 956 wheelbase (2650 mm)
        track_width=1.63,
        body_length=4.77,            # published 4770 mm overall
        body_width=1.99,             # published 1990 mm overall
        cg_height=0.30,              # ground-effect car, very low
        front_weight=0.42,           # rear-biased (mid-rear flat-6)
        yaw_inertia=1150.0,
        durability=0.4,              # aluminium monocoque prototype

        # --- tyres (1983 qualifying slicks) ---
        mu=1.95,                     # sticky qualifying rubber (> 787B race slicks)
        wheel_radius=0.33,
        wheel_inertia=0.9,
        load_sensitivity=2.0e-5,     # slicks handle load well
        pacejka_lat=Pacejka(B=12.0, C=1.6, E=0.95),
        pacejka_long=Pacejka(B=15.0, C=1.7, E=0.95),
        rolling_resistance=0.012,
        relaxation_length=0.20,
        relaxation_length_long=0.15,

        # --- handling balance ---
        roll_front_frac=0.46,
        steer_angle_max_deg=26.0,
        steer_rate_deg_s=500.0,
        offtrack_grip=0.5,

        # --- aero (FULL ground effect) ---
        drag_area=0.95,              # high-downforce Nürburgring setup
        side_drag_area=2.6,
        downforce_ClA=5.5,           # CALIBRATED: ~2.5x weight in downforce at speed
        aero_balance=0.42,

        # --- powertrain (Type 935 2.65 L twin-turbo flat-6) ---
        drivetrain_version="porsche956-5spd-ring-v1",
        engine_inertia=0.12,
        idle_rpm=2500.0,
        redline_rpm=8200.0,          # Type 935 flat-6 race limit
        cutoff_rpm=8200.0,
        engine_friction=0.08,
        idle_gain=10.0,
        # QUALIFYING boost trim (Bellof's 6:11.13 was a quali banzai): peak
        # ~498 kW = 677 PS near 6800 rpm — an ESTIMATE inside the documented
        # 620-750 PS range for wound-up 2.65 L 956 qualifying engines.
        torque_curve=[
            (2500, 240), (4000, 560), (5000, 670), (6000, 700),
            (6800, 700), (7600, 610), (8200, 540),
        ],
        gear_ratios=[2.83, 1.99, 1.55, 1.24, 0.92],
        final_drive=3.70,
        drivetrain_efficiency=0.95,
        clutch_capacity=1500.0,
        clutch_slip_ref=15.0,

        # --- twin-turbo ---
        boost_floor=0.45,            # notable off-boost lag (period turbo)
        spool_up_tau=0.40,
        spool_down_tau=0.12,

        # --- drivetrain layout ---
        drive_layout="rwd",
        lsd_coef=300.0,

        # --- suspension / visual ---
        body_roll_gain=0.15,
        suspension_tau=0.05,

        # --- brakes (ventilated steel — carbon came with the later 962) ---
        # 7000 Nm ≈ 2.6 g of pure brake authority (7000/0.33/820) — strong
        # 1983 steel discs, deliberately below the carbon-era cars.
        max_brake_torque=7000.0,
        brake_bias=0.58,
        handbrake_torque=0.0,
    )


def porsche_919_legacy() -> CarSpec:
    """The original legacy visually-simple 919 Evo."""
    return CarSpec(
        name="porsche_919_legacy",
        # --- chassis / mass ---
        mass=849.0,                  # Porsche-published Evo weight
        wheelbase=2.97,              # published 919 wheelbase (2970 mm) —
                                     # with the +1.7 m overhang footprint rule
                                     # the collision box is 4.67 m ≈ the real
                                     # 4.65 m overall length
        track_width=1.58,
        body_length=4.65,            # published 4650 mm overall
        body_width=1.90,             # published 1900 mm overall
        cg_height=0.33,              # extreme low LMP1 packaging
        weight_dist_front=0.46,

        # --- engine / battery / aero ---
        max_power=797000.0,
        max_torque=1400.0,
        battery_capacity=10.0,
        hybrid_boost_torque=400.0,
        max_rpm=9000.0,
        idle_rpm=1500.0,
        mu=2.10,
        downforce_ClA=6.5,
        drag_CdA=0.60,
        tire_radius=0.34,
        gears=[4.0, 3.2, 2.50, 2.05, 1.70, 1.45, 1.25, 1.10],
        final_drive=3.0,
        shift_time=0.03
    )

def porsche_919evo() -> CarSpec:
    """Legacy Porsche 919 Evo approximation for the lightweight 2D simulator.

    NONCERTIFIABLE: this preset predates faithful-v2. It combines public anchors
    and deliberately calibrated/inferred grip, aero, hybrid and gearing values;
    it does not separate the official 849 kg vehicle figure from driver/ballast
    and fuel. Preserve it for ordinary driving and historical playback only.
    It must never supply physics, oracle, evaluation, or certification evidence
    to the isolated faithful-v2 Porsche record program.

    Grip calibration note: mu (2.10) and downforce_ClA (6.5, ~3.3x weight at
    300 km/h) sit a little ABOVE the real car's measured peak aero. That is a
    deliberate modelling knob, not a claim about the physical car: the Fable
    envelope is a CENTERLINE reference and under-credits the racing line the
    real 919 actually drove, so grip is calibrated to REPRODUCE the known real
    result — centerline envelope ~372 s, racing line ~317 s — which brackets the
    old rounded 319.55 s display target. This target-fitting is specifically
    forbidden in faithful-v2.
    Gear ratios are optimizer-selected (see below); power/mass/top-speed match
    the real car (~370 km/h, ~1070 hp/tonne at the wheels).
    """
    return CarSpec(
        name="porsche_919evo",
        # --- chassis / mass ---
        mass=939.0,                  # 849 kg dry + 70 kg driver + 20 kg fluids
        wheelbase=2.97,              # published 919 wheelbase (2970 mm) —
                                     # with the +1.7 m overhang footprint rule
                                     # the collision box is 4.67 m ≈ the real
                                     # 4.65 m overall length
        track_width=1.58,
        body_length=4.65,            # published 4650 mm overall
        body_width=1.90,             # published 1900 mm overall
        cg_height=0.28,
        front_weight=0.46,
        yaw_inertia=1150.0,
        durability=0.4,

        # --- tyres (Michelin, unrestricted development) ---
        mu=2.10,                     # Realistic racing slick peak limit
        wheel_radius=0.355,          # Michelin 31/71-18: 710 mm diameter
        wheel_inertia=0.9,
        load_sensitivity=1.6e-5,
        pacejka_lat=Pacejka(B=13.0, C=1.6, E=0.95),
        pacejka_long=Pacejka(B=16.0, C=1.7, E=0.95),
        rolling_resistance=0.011,
        relaxation_length=0.18,
        relaxation_length_long=0.14,

        # --- handling balance ---
        roll_front_frac=0.48,
        steer_angle_max_deg=26.0,
        steer_rate_deg_s=520.0,
        offtrack_grip=0.5,

        # --- aero (active, unrestricted) ---
        # High-downforce (cornering) state. The Evo's DRS-style system dumps
        # wing drag on the straights: low-drag mode is drag_area * 0.86 =
        # 1.221 m^2, which with the 854 kW combined curve tops out at the
        # documented 369.4 km/h Doettinger Hoehe speed. ClA remains a
        # lap-time-calibrated value (no public Evo aero map exists).
        drag_area=1.42,
        side_drag_area=2.6,
        downforce_ClA=6.5,
        aero_active=True,
        aero_lowdrag_factor=0.84,      # CdA_low = 1.193 m^2
        aero_lowlift_factor=0.62,      # DRS dumps wing lift AND its induced
                                       # drag; with downforce-loaded rolling
                                       # resistance this tops out at ~368 km/h           # CALIBRATED (see note): ~3.3x weight @300km/h
        aero_balance=0.45,

        # --- powertrain (2.0 L V4 turbo + MGU, ~1160 hp combined) ---
        drivetrain_version="porsche919evo-7spd-ring-v2",
        engine_inertia=0.10,
        idle_rpm=3000.0,
        redline_rpm=9000.0,          # Porsche-specified ~9000 rpm limit
        cutoff_rpm=9000.0,
        engine_friction=0.07,
        idle_gain=10.0,
        # ICE-ONLY (720 PS) curve. The MGU is handled directly in physics.py
        # through the TTR hybrid logic based on speed and throttle.
        torque_curve=[
            (3000, 384), (4500, 558), (5500, 607), (6500, 620),
            (7500, 613), (8500, 595), (9000, 558),
        ],
        hybrid_mgu_power_w=324000.0,   # 440 PS MGU-K (published)
        hybrid_mgu_peak_torque_nm=550.0,  # CALIBRATED motor-torque cap. Base
                                       # speed ~35 m/s: below it the MGU is
                                       # torque-limited near the front grip
                                       # ceiling (939 kg, 46% front, μ 2.10
                                       # under launch weight transfer ≈ 460 Nm
                                       # motor); above it, full 324 kW power
                                       # for the 369 km/h top end. Prevents the
                                       # P/omega standstill torque explosion.
        hybrid_mguh_power_w=150000.0,  # ESTIMATED MGU-H exhaust harvest
        hybrid_ice_power_frac=1.0,     # Curve is now 100% ICE
        hybrid_battery_kj=8000.0,      # ESTIMATED usable pack energy (~2.2 kWh)
        hybrid_regen_eff=0.55,         # ESTIMATED brake-regen efficiency
        # Selected by tools/optimize_919_gearing.py (7-speed, retention band
        # 0.74–0.90, top gear ~0.94 redline at 98 m/s) and frozen as
        # supra/data/porsche_919evo_7spd_ring_v1.json.
        # Selected by tools/optimize_919_gearing.py (v2: 9000 rpm,
        # 0.355 m wheels, low-drag top-speed sizing); frozen as
        # supra/data/porsche_919evo_7spd_ring_v2.json.
        gear_ratios=[2.71311, 2.07847, 1.62674, 1.33554, 1.11466, 0.98215, 0.87296],
        final_drive=3.64479,
        drivetrain_efficiency=0.96,
        clutch_capacity=1800.0,
        clutch_slip_ref=15.0,

        # --- hybrid turbo (minimal lag) ---
        boost_floor=0.70,            # MGU fills torque off-boost
        spool_up_tau=0.20,
        spool_down_tau=0.10,

        # --- drivetrain layout (TTR Hybrid) ---
        drive_layout="awd",            # Kept for base compatibility
        is_ttr_hybrid=True,            # MGU front, ICE rear
        mgu_gear_ratio=5.5,            # Single speed reduction for MGU

        # --- suspension / visual ---
        body_roll_gain=0.12,
        suspension_tau=0.04,

        # --- brakes (carbon-carbon, brake-by-wire + regen) ---
        # 12500 Nm total ≈ 4.4 g of pure brake authority (12500/0.34/849 ≈
        # 43 m/s²); with aero drag the car peaks near 4.9 g at 300 km/h —
        # matching the ~5 g braking reported from the 919 Evo record runs.
        # The previous 6500 Nm allowed only 2.8 g, far below the real car and
        # 2.9x below what the tyre/downforce grip supports at speed.
        max_brake_torque=12500.0,
        brake_bias=0.56,
        handbrake_torque=0.0,
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
        durability=1.4,              # thick heavy chassis, strong AWD components
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
        durability=2.0,            # heavy 4x4, high structural armor
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
        durability=2.5,            # body-on-frame steel battering ram
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


PRESETS = {"supra": supra, "rx7": rx7, "skyline": skyline, "lr4": lr4, "f150": f150,
           "mazda787b": mazda787b, "porsche_956": porsche_956, "porsche_919evo": porsche_919evo,
           "porsche_919_legacy": porsche_919_legacy}


def get_car(name: str | CarSpec) -> CarSpec:
    if isinstance(name, CarSpec):
        return name
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
    fps: int = 120                   # fixed physics rate; V2 detects display refresh separately


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

    # --- pace block (Fable Five; obs 58 -> 58 + len(pace_distances) + 1) ---
    # Physics-true speed-envelope preview: v_ref at these distances down the
    # road (0 = here) + the current speed/envelope ratio. Requires a track
    # with an attached envelope (supra.fable5.attach_envelope); emits zeros
    # otherwise. Off by default — legacy obs layouts are untouched.
    pace_block: bool = False
    pace_distances: Tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 175.0, 275.0, 400.0)
    pace_ref: float = 90.0         # m/s normalization for the envelope speeds

    # --- hybrid block (fable-v2; obs +2) ---
    # Battery SOC (0..1) + signed MGU power / peak (+deploy, -regen), appended
    # AFTER the pace block so every fable-v1 index stays put. Off by default —
    # only hybrid-ledger cars (919 Evo) train with it; other cars emit the
    # frozen fable-v1 layout untouched.
    hybrid_block: bool = False


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
    action_log_std_max: Tuple[float, ...] | None = None
                                       # Optional per-action exploration ceilings.
                                       # Fable uses a much lower ceiling for its
                                       # rounded gear-offset action than for steer/
                                       # throttle, preventing random multi-gear hunts.
    target_kl: float | None = None     # trust region: stop an update's epoch loop
                                       # early once approx-KL(old||new) exceeds
                                       # 1.5x this. None = off (legacy behaviour).
                                       # Ratio clipping bounds the OBJECTIVE, not
                                       # how far 4 epochs x 8 minibatches can drag
                                       # the policy — near a knife-edge optimum
                                       # (Fable fast/frontier) one unguarded update
                                       # can turn a lap-capable brain into one that
                                       # spins at 20% of the lap.
    vf_clip: float | None = None       # PPO2-style value-loss clip: bound how far
                                       # V may move from the rollout's prediction
                                       # per update (max of clipped/unclipped MSE).
                                       # None = off (legacy). CAUTION: the clip is
                                       # in RAW return units — with unnormalised
                                       # returns O(100) (fable lap bonuses), 0.2
                                       # would freeze value learning; size it to
                                       # ~10% of the typical return scale if the
                                       # train log's vf loss starts spiking.

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
                                      # when the best score hasn't improved in this
                                      # many iters, trigger a plateau action (reseed
                                      # or stop — see max_restarts). Generous, so it
                                      # never bails during the early "learn-to-complete
                                      # then learn-to-drift" phase. 0 disables.
    max_restarts: int = 3             # on plateau, instead of stopping outright,
                                      # RESEED from the best checkpoint with a fresh
                                      # optimiser + re-widened exploration (a genuine
                                      # random-restart kick, not a plain continue) and
                                      # try again. Give up only after this many
                                      # CONSECUTIVE failed reseeds (the counter resets
                                      # whenever a reseed finds a new best). 0 = old
                                      # behaviour (stop immediately at patience).
    sensor_lookahead_distances: Tuple[float, ...] | None = None
                                      # Same number of samples as SensorSpec, but shifted
                                      # farther out for very long/high-speed real tracks
                                      # such as the Nordschleife. Obs dimensionality stays
                                      # identical; future checkpoints save the distances.
    sensor_pace_block: bool = False   # Fable Five: append the speed-envelope pace
                                      # block to the obs (SensorSpec.pace_block).
                                      # Changes obs size -> layout "fable-v1".
    sensor_pace_distances: Tuple[float, ...] | None = None
    sensor_hybrid_block: bool = False # fable-v2: append battery SOC + MGU power
                                      # to the obs (SensorSpec.hybrid_block).
                                      # Only hybrid-ledger cars (919 Evo).
    action_bias: Tuple[float, ...] | None = None
                                      # Per-action actor bias init override. Fable
                                      # uses (0, 0.6, 0): action[2] is a GEAR
                                      # OFFSET (neutral 0), not a handbrake (whose
                                      # default init is -1 = off).
    track_profile: str | None = None  # e.g. "nordschleife-full-20.832km"


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
    speed: float = 0.010              # per (speed / 80 m/s), per substep — baseline
                                      # "carry speed everywhere" pull (was 0.005, a
                                      # rounding error next to progress/align)
    offtrack: float = 0.15            # penalty per substep while off-track
    edge_band: float = 1.0            # metres of reward TAPER at the track edge: every
                                      # "doing well" reward (progress, align, speed, the
                                      # straight bonus, throttle commit) is scaled 1->0
                                      # over the outermost edge_band metres and is ZERO
                                      # off-track. Kills the exploit where the car rode
                                      # the grass/edge flat-out for speed+progress while
                                      # dodging the off-track timeout. Full reward across
                                      # the inner width, so real racing lines stay free.
    smooth: float = 0.0015            # penalty on action change^2 (was 0.002 — don't
                                      # over-punish the inputs needed to SEND it)
    crash: float = 3.0                # terminal penalty on crash
    lap_bonus: float = 5.0            # bonus for completing a lap

    # --- "SEND IT on the straights" shaping (push to the car's true top speed) ---
    # The plain speed/align/progress terms are all LINEAR in speed, so the slow
    # top-end (this car needs ~30 s of WOT to crawl 70 -> 80 m/s) earns barely more
    # than a safe cruise and the policy lifts early. These add a SUPER-LINEAR speed
    # reward that fires only where the road AHEAD is straight, plus a flat reward for
    # holding WOT there — so it floors long straights instead of settling.
    straight_speed: float = 0.06      # peak per-substep bonus, at top speed on a straight
    speed_ref: float = 82.0           # reward-side top-speed ref (car asymptotes ~82 m/s)
    speed_exp: float = 2.0            # super-linear: the LAST few m/s matter most
    straight_curv: float = 0.006      # |curv| (1/m) at/below which the road is fully
                                      # "straight" (~radius 167 m+) -> full bonus
    straight_span: float = 0.012      # curvature band over which the gate fades 1 -> 0
                                      # (by ~radius 56 m it's a corner -> no overspeed)
    throttle_commit: float = 0.02     # per-substep reward for full throttle when
                                      # straight + aligned (beats the 'lift early' habit
                                      # through the slow top-end of the straight)


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
    edge_band: float = 1.0
    smooth: float = 0.002
    crash: float = 3.0
    lap_bonus: float = 5.0
    straight_speed: float = 0.06
    speed_ref: float = 82.0
    speed_exp: float = 2.0
    straight_curv: float = 0.006
    straight_span: float = 0.012
    throttle_commit: float = 0.02
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
