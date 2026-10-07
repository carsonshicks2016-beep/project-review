"""Car presets.

``CarSpec`` is the **single tuning surface for handling**. Mass, weight
distribution, yaw inertia, Pacejka coefficients, relaxation length, differential
coefficients, torque curve, gearing, boost response and aero all live here and
nowhere else. If a number that changes how the car drives appears in the env, in
the track, or in a constants module, that is a bug — v1 had a ``constants.py``
with 72 knobs of which 9 were live and 7 silently duplicated the spec.
"""

from __future__ import annotations

from .vendor import CarSpec, Pacejka

# The surface friction that ``CarSpec.mu`` is quoted against: dry gravel.
#
# The vendored model computes peak grip as ``spec.mu * surface_grip``, so the
# tyre lives in the spec and the surface lives in the stage, combined by one
# multiplication. ``surface_grip = stage_mu / REFERENCE_SURFACE_MU``, which puts
# a car set up for gravel at 1.0, tarmac at ~1.36, and snow at ~0.5.
REFERENCE_SURFACE_MU = 0.72


def surface_grip(stage_mu: float) -> float:
    """Convert a stage's surface friction into the vendored model's multiplier."""
    return float(stage_mu) / REFERENCE_SURFACE_MU


def evo_rally() -> CarSpec:
    """Group-A / WRC-era Mitsubishi Evo.

    Turbocharged inline-four, AWD with a locking centre differential, Group A
    minimum weight. Set up to be snappier and more nervous than a heavier,
    longer-wheelbase rally car: less yaw inertia, shorter tyre relaxation length
    so grip arrives and leaves faster, and rear-biased roll stiffness so it
    rotates on entry. It should reward commitment and punish laziness.

    These are feel targets, not measured truth. They are a starting point to be
    tuned against the §4 table by driving it, and every one of them is a
    deliberate choice rather than a default — but none of them is sacred.
    """
    return CarSpec(
        name="evo_rally",

        # --- chassis -------------------------------------------------------
        # Group A minimum weight. Short wheelbase and low yaw inertia are what
        # make it point; a heavier, longer car is calmer and slower to rotate.
        mass=1250.0,
        wheelbase=2.51,
        track_width=1.47,
        cg_height=0.49,
        front_weight=0.58,          # transverse 4G63 sits over the front axle
        yaw_inertia=1850.0,         # low: this is the "snappy" dial
        durability=1.40,            # rally cars are built to be hit

        # --- tyres ---------------------------------------------------------
        # mu is quoted against REFERENCE_SURFACE_MU (dry gravel).
        mu=1.05,
        wheel_radius=0.32,
        wheel_inertia=1.30,
        load_sensitivity=5.5e-5,
        pacejka_lat=Pacejka(B=8.8, C=1.52, E=0.94),
        pacejka_long=Pacejka(B=12.8, C=1.62, E=0.88),
        rolling_resistance=0.018,
        # Shorter than a tarmac car's: grip builds and collapses quickly, which
        # is what makes a slide catchable but unforgiving of a late correction.
        relaxation_length=0.42,
        relaxation_length_long=0.20,

        # --- handling ------------------------------------------------------
        roll_front_frac=0.46,       # rear-biased roll stiffness -> rotation
        steer_angle_max_deg=38.0,   # rally cars run a lot of lock
        steer_rate_deg_s=340.0,
        offtrack_grip=0.55,

        # --- aero (upright, draggy, barely any downforce) -------------------
        drag_area=0.74,
        side_drag_area=2.3,         # big: sideways scrubs speed hard
        aero_balance=0.42,

        # --- 4G63 turbo -----------------------------------------------------
        engine_inertia=0.30,
        idle_rpm=950.0,
        redline_rpm=7600.0,
        cutoff_rpm=7900.0,
        engine_friction=0.036,
        torque_curve=[
            (950, 210),
            (2000, 340),
            (3000, 455),
            (4000, 490),
            (5000, 480),
            (6000, 435),
            (7000, 375),
            (7600, 315),
            (7900, 0),
        ],
        # Group A close-ratio five-speed with a short gravel final drive.
        # Rally cars are gear-limited, not drag-limited: 5th runs out at about
        # 205 km/h, and 1st tops out near 70 km/h so a hairpin can be taken in
        # gear. A longer final drive gave 234 km/h, which is a tarmac ratio.
        gear_ratios=[2.79, 1.95, 1.44, 1.10, 0.86],
        final_drive=5.20,
        drivetrain_efficiency=0.88,
        clutch_capacity=900.0,
        # High boost floor and a fast spool stand in for anti-lag: the throttle
        # answers immediately, which is most of why the car feels alive.
        boost_floor=0.62,
        spool_up_tau=0.24,
        spool_down_tau=0.14,

        # --- AWD with a locking centre diff ---------------------------------
        drive_layout="awd",
        center_split_front=0.50,
        center_diff="lsd",
        center_lsd_coef=190.0,
        lsd_coef=150.0,
        front_lsd_coef=90.0,
        coupling="clutch",

        # --- brakes and handbrake -------------------------------------------
        max_brake_torque=2600.0,
        brake_bias=0.60,
        # Strong enough to lock the rears and pivot a hairpin. The handbrake is
        # not a garnish — it is the single most recognisable thing in the sport,
        # and it is in the action space.
        handbrake_torque=4200.0,
        suspension_tau=0.16,

        # --- body (collision + rendering) ------------------------------------
        body_length=4.35,
        body_width=1.77,
    )


PRESETS = {
    "evo_rally": evo_rally,
}


def get(name: str) -> CarSpec:
    if name not in PRESETS:
        raise KeyError(f"unknown car {name!r}; known: {', '.join(sorted(PRESETS))}")
    return PRESETS[name]()
