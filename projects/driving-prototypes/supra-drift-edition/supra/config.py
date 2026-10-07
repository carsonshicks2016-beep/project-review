"""
Central configuration for the Supra-AI simulation.

Everything is in SI units (metres, kilograms, seconds, Newtons, radians)
unless noted.  Numbers are modelled on the Mk4 (A80) Toyota Supra with the
2JZ-GTE twin-turbo inline-six.  Tweak freely -- the physics reads from here.
"""

from dataclasses import dataclass, field
import math


# --------------------------------------------------------------------------
# CAR -- Mk4 Supra (A80) / 2JZ-GTE
# --------------------------------------------------------------------------
@dataclass
class CarSpec:
    name: str = "Toyota Supra Mk4 (A80) 2JZ-GTE"
    engine: str = "2JZ-GTE"        # short label for the dashboard

    # ---- Mass & geometry ----
    mass: float = 1540.0           # kg, curb-ish
    wheelbase: float = 2.55        # m
    # Weight distribution ~53% front / 47% rear (front-engine RWD).
    # front axle load fraction = lr / wheelbase, so a heavier front => CG nearer front.
    front_weight_frac: float = 0.53
    cg_height: float = 0.46        # m, centre of gravity height
    track_width: float = 1.55      # m, left-to-right wheel spacing (for visuals)
    body_length: float = 4.51      # m
    body_width: float = 1.81       # m

    # Yaw inertia (kg*m^2). Iz ~ mass * lf * lr is a decent approximation.
    @property
    def lf(self) -> float:         # CG -> front axle
        return self.wheelbase * (1.0 - self.front_weight_frac)

    @property
    def lr(self) -> float:         # CG -> rear axle
        return self.wheelbase * self.front_weight_frac

    @property
    def yaw_inertia(self) -> float:
        return self.mass * self.lf * self.lr

    # ---- Wheels / tyres ----
    wheel_radius: float = 0.33     # m (≈ 245/40R18)
    wheel_inertia: float = 1.4     # kg*m^2 per wheel (rotational)
    tyre_mu: float = 1.15          # peak friction coefficient (sport tyre on dry tarmac)
    # Simplified Pacejka "magic formula": F = D sin(C atan(B s - E(B s - atan(B s))))
    pacejka_B: float = 9.5         # lateral stiffness
    pacejka_C: float = 1.45        # lateral shape
    pacejka_E: float = 0.97        # lateral curvature
    pacejka_B_long: float = 14.0   # longitudinal (slip-ratio) stiffness
    pacejka_C_long: float = 1.65
    pacejka_E_long: float = 0.90
    # Load sensitivity: peak grip coefficient drops as a tyre is overloaded,
    # so weight transfer reduces *total* grip -> rewards smoothness.
    load_sensitivity: float = 0.20
    # Fraction of lateral weight transfer reacted by the FRONT axle (roll
    # stiffness balance). Higher -> more understeer.
    roll_stiffness_front: float = 0.55
    pneumatic_trail: float = 0.10  # m, gives self-aligning torque (steering feel)

    # ---- Aerodynamics & resistance ----
    drag_cd: float = 0.32
    frontal_area: float = 1.92     # m^2
    air_density: float = 1.225     # kg/m^3
    rolling_resistance: float = 0.015

    # ---- Engine: 2JZ-GTE ----
    idle_rpm: float = 850.0
    redline_rpm: float = 6800.0
    rev_limit_rpm: float = 7000.0
    # Fully-spooled torque curve (rpm -> Nm).  Already represents a boosted
    # engine; the turbo model below scales this down when off-boost.
    torque_curve: tuple = (
        (850, 150), (1500, 285), (2000, 345), (2500, 400),
        (3000, 440), (3600, 462), (4500, 452), (5500, 410),
        (6000, 380), (6800, 330), (7000, 300),
    )
    engine_brake_torque: float = 45.0   # Nm of drag at closed throttle
    drivetrain_efficiency: float = 0.90
    engine_inertia: float = 0.30        # kg*m^2 flywheel + rotating assembly

    # ---- Clutch (brain-controllable) ----
    clutch_max_torque: float = 800.0    # Nm capacity (>= peak engine torque)
    clutch_slip_scale: float = 12.0     # rad/s slip for ~full transmitted torque

    # ---- Sequential twin-turbo ----
    max_boost_bar: float = 0.80         # ~11.6 psi, stock-ish
    boost_spool_start_rpm: float = 1800.0
    boost_full_rpm: float = 4000.0
    boost_spool_tau: float = 0.28       # s, lag building boost
    boost_decay_tau: float = 0.12       # s, faster bleed-down off throttle
    na_torque_fraction: float = 0.42    # torque available at zero boost

    # ---- Gearbox: Getrag V160 6-speed ----
    gear_ratios: tuple = (3.827, 2.360, 1.685, 1.312, 1.000, 0.793)
    final_drive: float = 3.266
    shift_up_rpm: float = 6400.0
    shift_down_rpm: float = 2600.0
    shift_time: float = 0.18            # s of torque interruption per shift

    # ---- Limited-slip differential (rear) ----
    lsd_lock: float = 40.0              # Nm per rad/s of rear wheel-speed diff

    # ---- Suspension (transient load transfer) ----
    suspension_tau: float = 0.14       # s, time for weight to settle after a load change
    roll_gain_deg: float = 6.0         # body roll per g (visual / telemetry)
    pitch_gain_deg: float = 3.0        # body pitch per g (visual / telemetry)

    # ---- Brakes & steering ----
    max_brake_g: float = 1.15           # peak braking deceleration in g
    brake_bias_front: float = 0.62      # fraction of brake torque to front axle
    max_steer_angle: float = math.radians(30.0)
    steer_rate: float = math.radians(220.0)   # rad/s steering actuator speed
    # Human-style actuator rates (full pedal travel per second).
    throttle_rate: float = 6.0
    brake_rate: float = 8.0
    # Handbrake: extra REAR-only brake torque (Nm) when pulled.  Locks the rears
    # to break traction -> instant oversteer.  Used by the drift policy (the
    # racer leaves it at 0).
    handbrake_torque: float = 3600.0


# --------------------------------------------------------------------------
# SENSORS / BRAIN INTERFACE
# --------------------------------------------------------------------------
@dataclass
class SensorSpec:
    n_rays: int = 9                       # raycast "vision" beams
    ray_spread: float = math.radians(100) # +/- half-angle covered by the fan
    ray_max_range: float = 70.0           # m
    # Extra proprioceptive inputs appended after the rays:
    #   vx, vy, yaw_rate, slip_angle, rpm_norm, gear_norm, lat_g, prev_steer,
    #   self_aligning_torque (steering feel)
    n_extra: int = 9
    # Look-ahead "track preview": signed centreline curvature (left +, right -)
    # sampled at these arc-distances (m) ahead of the car.  This lets a
    # feed-forward policy *plan* braking points and apexes -- the key to a fast
    # lap -- instead of only reacting to what the rays currently see.
    curve_preview_dists: tuple = (8.0, 16.0, 28.0, 44.0, 64.0, 90.0)

    @property
    def n_curve(self) -> int:
        return len(self.curve_preview_dists)

    @property
    def n_inputs(self) -> int:
        return self.n_rays + self.n_extra + self.n_curve

    # steer, throttle, brake, clutch, shift_up, shift_down
    n_outputs: int = 6


# --------------------------------------------------------------------------
# TRACK
# --------------------------------------------------------------------------
@dataclass
class TrackSpec:
    n_control_points: int = 14
    base_radius: float = 220.0     # m
    radius_jitter: float = 90.0    # m of random variation per control point
    samples: int = 720             # centreline resolution
    half_width: float = 11.0       # m (track is 22 m wide -- roomy)
    min_corner_radius: float = 28.0

    # ---- Touge / mountain pass ----
    touge_descent: float = 120.0       # metres of total elevation drop
    touge_half_width: float = 7.0      # narrower than circuit (11m default)
    touge_hairpins: int = 6            # number of switchback turns
    touge_grade_max: float = 0.12      # maximum road gradient (rise/run)


# --------------------------------------------------------------------------
# EVOLUTION (genetic algorithm)
# --------------------------------------------------------------------------
@dataclass
class EvoSpec:
    population: int = 24
    hidden_layers: tuple = (24, 18)   # MLP hidden sizes
    elite_fraction: float = 0.15      # top fraction copied unchanged
    survivor_fraction: float = 0.45   # top fraction allowed to breed
    mutation_rate: float = 0.18       # prob. a weight is perturbed
    mutation_scale: float = 0.30      # std-dev of perturbation
    weight_init_scale: float = 0.8
    # A generation ends when the leader has proven itself (completed this many
    # laps), OR everyone has stalled, OR the time cap hits -- whichever first.
    # The lap cap keeps the GENERATION ticker moving at a watchable pace.
    gen_max_laps: int = 2               # a completed lap ends the gen (natural milestone)
    max_episode_seconds: float = 120.0  # hard ceiling -- enough to lap a big track
    # End early once no car has improved its best distance for this long
    # (i.e. everyone is dead or hopelessly stuck).
    gen_stagnation_seconds: float = 4.0
    idle_kill_seconds: float = 3.0    # an individual car is killed after this long with no progress


# --------------------------------------------------------------------------
# SIM / TIMING
# --------------------------------------------------------------------------
@dataclass
class SimSpec:
    dt: float = 1.0 / 60.0
    physics_substeps: int = 4         # integrate physics N times per frame for stability
    drivetrain_substeps: int = 6      # extra sub-stepping for the stiff engine/clutch/wheel dynamics
    gravity: float = 9.81


# --------------------------------------------------------------------------
# CONTROL MODE
# --------------------------------------------------------------------------
@dataclass
class ControlSpec:
    # When True the brain drives the gearbox/clutch itself; when False the car
    # auto-shifts and keeps the clutch engaged (the simpler legacy behaviour).
    manual_gears: bool = True
    manual_clutch: bool = True


# --------------------------------------------------------------------------
# CURRICULUM / TRACK RANDOMISATION
# --------------------------------------------------------------------------
@dataclass
class CurriculumSpec:
    enabled: bool = True
    # GA: rotate to a fresh track every N generations (fair within a gen).
    ga_rotate_generations: int = 2
    # PPO: each finished episode reports how much of a lap it covered
    # (travelled / track_length, capped at `promote_fraction_cap`).  Once the
    # mean completion fraction over the last `promote_window` episodes reaches
    # `promote_threshold`, the trainer unlocks the next (harder) track tier.
    # (Was "fraction of episodes that completed a WHOLE lap" — unreachable on
    # 2.5-3.5 km procedural tracks within the time budget, so it never promoted.)
    promote_window: int = 80
    promote_threshold: float = 0.6     # mean lap-fraction needed to promote
    promote_fraction_cap: float = 2.0  # cap a single episode's reported fraction
    start_stage: int = 0


# --------------------------------------------------------------------------
# RENDER
# --------------------------------------------------------------------------
@dataclass
class RenderSpec:
    width: int = 1480
    height: int = 920
    dash_width: int = 440             # right-hand telemetry panel
    fps: int = 60
    bg_color: tuple = (22, 26, 24)            # deep base behind the grass
    grass_color: tuple = (34, 46, 38)         # infield / runoff
    grass_color2: tuple = (30, 41, 34)        # subtle stripe
    track_color: tuple = (54, 56, 62)         # asphalt
    track_color2: tuple = (49, 51, 57)        # asphalt stripe
    track_edge: tuple = (220, 224, 230)       # white line
    kerb_color1: tuple = (210, 70, 64)        # red/white kerb
    kerb_color2: tuple = (235, 238, 244)
    centerline: tuple = (120, 120, 80)
    supra_color: tuple = (210, 40, 38)        # Renaissance Red-ish
    focus_color: tuple = (255, 215, 60)
    smoke_color: tuple = (200, 200, 205)
    meters_per_pixel_default: float = 0.26


# --------------------------------------------------------------------------
# PPO -- reinforcement learning (the path to a genuinely fast lap)
# --------------------------------------------------------------------------
@dataclass
class PPOSpec:
    """Everything the PPO trainer reads.

    The crucial design choice lives here: episodes run on a fixed *time budget*
    and the reward is *centre-line progress per step*.  Maximising the return
    therefore means "cover the most track in the time you have" == drive FAST.
    (The old setup ended every episode after a fixed number of laps, so the
    return was a near-constant distance and there was no incentive to be quick.)
    """
    # ---- network ----
    hidden: tuple = (128, 128)
    # Opt-in recurrent (LSTM) policy: an encoder MLP -> LSTM -> heads, with memory
    # across timesteps for anticipation / smoother trail-braking.  Off by default
    # (the feed-forward net is the proven path); enable with --recurrent.
    recurrent: bool = False
    lstm_hidden: int = 128             # LSTM cell size when recurrent
    recurrent_minibatch_envs: int = 8  # sequences (envs) per PPO minibatch

    # ---- rollout / parallelism ----
    n_envs: int = 24            # parallel tracks per update (more = more diverse
                                # gradient & better generalisation; costs CPU).
                                # Override at runtime with --n-envs.
    n_workers: int = 0          # 0 = auto (cpu_count-1, capped); 1 = in-process
    horizon: int = 512          # env steps gathered per env per PPO iteration

    # ---- optimisation ----
    lr: float = 3.0e-4
    lr_final_frac: float = 0.1  # linearly anneal LR to this fraction by the end
    gamma: float = 0.997        # long credit assignment -> anticipate/brake for corners
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 2048
    # Exploration: keep this LOW.  The action std (log_std) is hard-clamped in
    # ppo.py (LOG_STD_MIN/MAX); a high ent_coef holds std at the ceiling so the
    # policy never sharpens into a precise racing line.  0.004 lets the policy
    # gradient pull std down toward precision while retaining some exploration.
    ent_coef: float = 0.004
    ent_final_frac: float = 0.3  # anneal ent_coef to this fraction by run's end
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03     # stop an iteration's epochs early past this KL
    obs_clip: float = 5.0       # clip normalised observations to +/- this

    # ---- episode ----
    # TIME budget -> truncation (bootstrapped).  Must be long enough that a
    # competent policy can complete a meaningful fraction of a lap on the
    # current-tier tracks, or the curriculum can never promote.  At ~18 m/s a
    # 130 s episode covers ~2.3 km (most of a stage-0/1 lap).
    max_episode_seconds: float = 130.0
    idle_kill_seconds: float = 3.0     # no centre-line progress this long = fail

    # ---- reward shaping (per physics frame) ----
    progress_weight: float = 1.0    # metres of centre-line progress  (dominant)
    speed_weight: float = 0.02      # small carry-speed bonus (per second)
    jerk_weight: float = 0.02       # penalise sawing the controls (smoothness)
    time_cost: float = 0.0          # optional per-step living cost
    crash_penalty: float = 25.0     # leaving the track ends the episode (-this);
                                    # big enough that over-speeding into a corner
                                    # is clearly not worth it -> learns to brake

    # ---- DRIFT objective (separate \"drift lineage\"; train with --drift) ----
    # Purely multiplicative: reward = speed × angle × sustain.
    # ZERO reward from driving without drifting.  Must go fast AND sideways.
    drift_weight: float = 2.0          # core: speed × sin(slip) weight (high = dominant)
    drift_progress_weight: float = 0.02  # near-zero: just enough to break donut ties
    drift_min_angle: float = 0.22      # rad ~13°: entry threshold for drift detection
    drift_cap_angle: float = 1.18      # rad ~67°: sin() reward caps here
    drift_spin_angle: float = 1.40     # rad ~80°: beyond = spin-out (episode ends)
    drift_spin_penalty: float = 3.0    # mild: don't terrorize the agent for pushing limits
    drift_min_speed: float = 12.0      # m/s (~43 km/h): must be moving fast to earn
    drift_angle_bonus: float = 1.2     # quadratic angle scaling (bigger slides = way more pay)
    drift_speed_bonus: float = 0.5     # continuous speed reward during drifts (no ceiling)
    drift_speed_shaping: float = 0.05  # always-on speed reward (curriculum: learn to drive fast first)
    drift_slow_penalty: float = 0.15   # gentle per-second cost below drift_min_speed
    drift_sustain_bonus: float = 0.25  # per-second sustain multiplier ramp
    drift_sustain_max: float = 3.0     # seconds: sustain caps at +75% (0.25 × 3.0)
    crash_penalty_drift: float = 12.0  # lower than race (25): encourage limit-pushing

    # ---- Drift chain / transition scoring ----
    drift_chain_bonus: float = 0.6       # per-frame bonus per chain link (raised)
    drift_chain_transition_window: float = 1.5  # max seconds between drifts to keep chain
    drift_chain_min_speed: float = 10.0  # speed floor during transitions to keep chain
    # ---- Gutter mechanic (touge) ----
    gutter_drift_bonus: float = 0.3      # bonus for clipping gutter while drifting
    gutter_grip_penalty: float = 0.1     # penalty for clipping gutter while gripping

    # ---- Drift overhaul: make the reward learnable + un-trap + give it tools ----
    drift_init_weight: float = 0.35      # reward for breaking rear traction (wheelspin) at speed
    drift_entry_lo_angle: float = 0.12   # rad ~7deg: soft window starts here (gradient INTO a slide)
    drift_ent_coef: float = 0.02         # extra exploration for the drift policy
    drift_steer_lock_deg: float = 48.0   # the drift car gets more lock (hold big angles)
    drift_steer_rate_deg: float = 320.0  # ...and quicker hands for transitions
    drift_stall_seconds: float = 3.0     # window for the donut/stall check
    drift_stall_displacement: float = 7.0  # m: < this net travel in the window = donut/stall (terminal)
    # Drift trains on cornery, alternating-corner tracks so chaining can fire.
    drift_track_pool: tuple = ("loop", "technical")
    # Drift gets its OWN episode budget so the racer's max_episode_seconds can't
    # bleed in (the two lineages stay fully independent).
    drift_max_episode_seconds: float = 120.0


@dataclass
class Config:
    car: CarSpec = field(default_factory=CarSpec)
    sensors: SensorSpec = field(default_factory=SensorSpec)
    track: TrackSpec = field(default_factory=TrackSpec)
    evo: EvoSpec = field(default_factory=EvoSpec)
    sim: SimSpec = field(default_factory=SimSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    curriculum: CurriculumSpec = field(default_factory=CurriculumSpec)
    ppo: PPOSpec = field(default_factory=PPOSpec)


# --------------------------------------------------------------------------
# CAR PRESETS  --  swap the chassis with --car {supra,rx7,skyline}
# --------------------------------------------------------------------------
# Game-feel approximations, not data-sheet exact.  NOTE: the physics model is
# rear-wheel-drive, so the (really AWD) Skyline is a RWD approximation -- its
# character comes from mass, power, balance and gearing.  Same tyre/Pacejka
# model across all three; what differs is the stuff that changes how it drives.

def _supra() -> CarSpec:
    return CarSpec()      # the default = Mk4 Supra / 2JZ-GTE


def _rx7() -> CarSpec:
    """Mazda RX-7 FD3S (13B-REW): light, ~50/50, high-revving rotary, peaky."""
    return CarSpec(
        name="Mazda RX-7 FD3S", engine="13B-REW",
        mass=1310.0, wheelbase=2.425, front_weight_frac=0.505,
        cg_height=0.45, track_width=1.50, body_length=4.30, body_width=1.76,
        wheel_radius=0.31, wheel_inertia=1.1, tyre_mu=1.12,
        idle_rpm=900.0, redline_rpm=8000.0, rev_limit_rpm=8200.0,
        torque_curve=((1000, 150), (2000, 205), (3000, 250), (4000, 283),
                      (5000, 294), (6000, 285), (7000, 255), (8000, 215), (8200, 180)),
        engine_brake_torque=38.0, engine_inertia=0.18,
        max_boost_bar=0.70, boost_spool_start_rpm=2000.0, boost_full_rpm=4200.0,
        na_torque_fraction=0.40,
        gear_ratios=(3.483, 2.015, 1.391, 1.000, 0.762), final_drive=4.100,
        shift_up_rpm=7600.0, shift_down_rpm=3000.0,
        lsd_lock=35.0, max_brake_g=1.15, brake_bias_front=0.62)


def _skyline() -> CarSpec:
    """Nissan Skyline GT-R R34 (RB26DETT): heavy, front-biased, torquey, grippy
    (modelled RWD)."""
    return CarSpec(
        name="Nissan Skyline GT-R R34", engine="RB26DETT (RWD sim)",
        mass=1560.0, wheelbase=2.665, front_weight_frac=0.56,
        cg_height=0.47, track_width=1.48, body_length=4.60, body_width=1.785,
        wheel_radius=0.33, wheel_inertia=1.5, tyre_mu=1.18,
        idle_rpm=850.0, redline_rpm=7200.0, rev_limit_rpm=7600.0,
        torque_curve=((1000, 200), (2000, 330), (3000, 378), (4000, 392),
                      (4400, 392), (5000, 378), (6000, 350), (7000, 310), (7600, 260)),
        engine_brake_torque=46.0, engine_inertia=0.34,
        max_boost_bar=0.85, boost_spool_start_rpm=2600.0, boost_full_rpm=4400.0,
        na_torque_fraction=0.45,
        gear_ratios=(3.827, 2.360, 1.685, 1.312, 1.000, 0.793), final_drive=3.545,
        shift_up_rpm=6900.0, shift_down_rpm=2800.0,
        lsd_lock=50.0, max_brake_g=1.18, brake_bias_front=0.60)


CARS = {"supra": _supra, "rx7": _rx7, "skyline": _skyline}


def car_spec(name="supra") -> CarSpec:
    """Return a fresh CarSpec for the named preset (defaults to the Supra)."""
    return CARS.get(name, _supra)()


# A ready-to-use default instance.
CFG = Config()
