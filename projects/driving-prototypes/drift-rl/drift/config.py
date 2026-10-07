"""Vehicle, arena, scoring and curriculum configuration for the drift trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

DEG = math.pi / 180.0


@dataclass(frozen=True)
class TireParams:
    """Magic-formula shape coefficients for one axle (peak force is mu * Fz)."""

    B: float = 9.5          # stiffness
    C: float = 1.55         # shape
    E: float = 0.97         # curvature
    mu: float = 1.15        # peak friction coefficient


@dataclass(frozen=True)
class CarConfig:
    """A rear-drive coupe with a deliberately loose back end."""

    mass: float = 1500.0
    inertia: float = 2300.0          # kg m^2 about the vertical axis
    a: float = 1.25                  # m, CG to front axle
    b: float = 1.45                  # m, CG to rear axle
    cg_height: float = 0.52
    track_width: float = 1.58        # only used for drawing
    body_len: float = 4.4
    body_wid: float = 1.85

    wheel_radius: float = 0.32
    wheel_inertia: float = 2.6       # kg m^2, both rear wheels + driveline

    # Rear tyres give up slightly sooner than the fronts, which is what makes
    # the car rotate instead of push when you lean on it.
    front_tire: TireParams = field(default_factory=lambda: TireParams(B=11.0, C=1.62, mu=1.22))
    rear_tire: TireParams = field(default_factory=lambda: TireParams(B=8.5, C=1.50, mu=1.05))

    max_steer: float = 50.0 * DEG    # angle-kit lock; you need it past 40 deg of slip
    steer_rate: float = 4.2          # rad/s at the roadwheel (fast hands)

    drive_torque: float = 3900.0     # N m at the rear wheels
    drive_power: float = 300e3       # W, caps torque once rolling
    brake_torque: float = 3200.0     # N m, rear-biased on purpose
    handbrake_torque: float = 9000.0

    drag_coef: float = 0.42          # 0.5 * rho * Cd * A
    roll_resist: float = 0.015

    dt: float = 0.04                 # s per control step (25 Hz)
    substeps: int = 16               # physics runs at 400 Hz for wheel-slip stability


@dataclass(frozen=True)
class ArenaConfig:
    radius: float = 70.0             # m, circular wall
    pillar_radius: float = 0.9       # m, the cones/poles you clip
    clip_band: float = 4.5           # m from the pillar surface counts as a clip
    n_beams: int = 9                 # rangefinder rays for wall/pillar sensing
    beam_range: float = 60.0


@dataclass(frozen=True)
class ScoreConfig:
    """Arcade scoring: angle x speed x time, multiplied by whatever you chain."""

    angle_min: float = 12.0 * DEG    # rear slip needed to be 'drifting'
    speed_min: float = 7.0           # m/s
    grace: float = 0.45              # s you may straighten before the combo banks

    base_rate: float = 0.075         # points per (deg of slip * m/s * s)
    # The multiplier has two budgets. Time alone saturates fast and low, so
    # parking in one endless drift caps out; only linking tricks opens up the
    # top of the scale. This is the whole reason the task is about chaining.
    mult_per_second: float = 0.20    # growth per second of live combo
    mult_time_cap: float = 2.0       # ...but time contributes at most this much
    mult_max: float = 12.0

    # Trick bonuses (flat points, then scaled by the live multiplier).
    switchback_bonus: float = 90.0
    switchback_mult: float = 0.5
    # A switchback only counts if the previous slide was a real one, otherwise
    # a policy can farm the bonus by wiggling the tail across zero.
    switchback_min_angle: float = 20.0 * DEG
    switchback_min_time: float = 0.35
    donut_bonus: float = 160.0
    donut_mult: float = 0.8
    clip_bonus: float = 220.0
    clip_mult: float = 0.75
    manji_bonus: float = 260.0       # three switchbacks inside manji_window
    manji_window: float = 4.0
    wall_kiss_bonus: float = 120.0   # drifting within wall_kiss_dist of the wall
    wall_kiss_dist: float = 2.5
    wall_kiss_cooldown: float = 1.5
    big_angle: float = 45.0 * DEG    # sustained huge angle
    big_angle_hold: float = 0.6
    big_angle_bonus: float = 140.0

    spin_angle: float = 105.0 * DEG  # past this the car is a passenger
    spin_speed: float = 3.0          # dropping below this mid-combo is a bog-down
    spin_pending_frac: float = 0.5   # fraction of the un-banked combo you forfeit


@dataclass(frozen=True)
class RewardConfig:
    score_scale: float = 0.1         # points -> reward; keeps a decent run
    #                                worth several crashes, so the policy is
    #                                pushed to take risk rather than survive
    crash_penalty: float = 25.0
    spin_penalty: float = 8.0
    stall_penalty: float = 4.0       # the real cost of stalling is the score you forgo
    # Scoring only starts past 12 deg of slip, which random exploration will
    # basically never reach and hold. These two terms are a smooth ramp into
    # that regime; both are an order of magnitude below the score itself once
    # the car is actually drifting, so they guide without dictating the style.
    speed_shaping: float = 0.015     # per (m/s * s)
    slide_shaping: float = 0.06      # per (rad of slip * m/s * s), ungated
    pillar_shaping: float = 0.15     # potential-based pull toward the next clip


@dataclass(frozen=True)
class Stage:
    """One rung of the curriculum."""

    name: str
    n_pillars: int
    arena_radius: float
    episode_seconds: float
    spawn_speed: tuple[float, float]
    promote_at: float                # mean normalised score needed to advance


CURRICULUM: tuple[Stage, ...] = (
    Stage("slide",     0, 65.0, 22.0, (13.0, 20.0), 0.35),
    Stage("orbit",     1, 65.0, 28.0, (12.0, 20.0), 0.35),
    Stage("figure8",   2, 70.0, 34.0, (12.0, 20.0), 0.32),
    Stage("gymkhana",  4, 80.0, 40.0, (10.0, 20.0), 0.30),
)


def stage_arena(base: ArenaConfig, stage: Stage) -> ArenaConfig:
    return replace(base, radius=stage.arena_radius)


def stage_by_name(name: str) -> Stage:
    for s in CURRICULUM:
        if s.name == name:
            return s
    raise KeyError(f"unknown stage {name!r}; have {[s.name for s in CURRICULUM]}")
