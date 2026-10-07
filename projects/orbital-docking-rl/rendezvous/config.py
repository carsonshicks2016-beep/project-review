"""Task, vehicle and curriculum configuration for the docking environment."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ShipConfig:
    max_accel: float = 0.012         # m/s^2 at full throttle; one step of thrust
    #                                changes velocity by ~0.024 m/s, comfortably
    #                                finer than the capture speed tolerance
    dv_budget: float = 12.0          # m/s of delta-v carried for the whole episode
    min_throttle: float = 0.02       # commands below this are treated as off (deadband)


@dataclass(frozen=True)
class TaskConfig:
    altitude: float = 420e3          # m, ISS-like circular orbit
    dt: float = 2.0                  # s per control step
    max_steps: int = 600             # 20 minutes of proximity ops

    # Approach corridor: a cone about the target's docking-port axis (+V-bar).
    port_axis: tuple = (0.0, 1.0, 0.0)
    cone_half_angle_deg: float = 25.0
    keepout_radius: float = 20.0     # m; must be inside the cone to come closer

    # Capture conditions.
    dock_radius: float = 1.0         # m
    dock_speed: float = 0.12         # m/s total relative speed
    dock_lateral_speed: float = 0.04  # m/s perpendicular to the port axis

    # Safety envelope: |v| <= v_cap_a + v_cap_b * range
    v_cap_a: float = 0.08
    v_cap_b: float = 0.012
    bounds: float = 900.0           # m; leaving this sphere aborts the episode

    # Reward weights.
    w_range: float = 1.0             # shaping on range
    w_speed: float = 10.0             # shaping on speed
    w_fuel: float = 0.6              # per m/s of delta-v spent
    w_vcap: float = 2.0              # per (m/s) over the safety envelope
    w_cone: float = 1.5              # off-axis penalty inside the keep-out sphere
    r_dock: float = 250.0            # capture bonus
    r_crash: float = -120.0          # keep-out violation / hard contact
    r_abort: float = -60.0           # out of bounds or out of fuel

    # Observation normalisation.
    len_scale: float = 150.0
    vel_scale: float = 1.0


@dataclass(frozen=True)
class Stage:
    """One rung of the curriculum."""

    name: str
    range_min: float                 # m, initial separation
    range_max: float
    vel_sigma: float                 # m/s, initial relative velocity spread
    spawn_cone_deg: float            # spawn direction spread about the port axis
    promote_at: float                # success rate needed to advance


CURRICULUM = (
    Stage("close-hold",   10.0,   30.0, 0.02,  35.0, 0.85),
    Stage("corridor",     25.0,   70.0, 0.04,  70.0, 0.80),
    Stage("mid-range",     60.0, 180.0, 0.08, 120.0, 0.75),
    Stage("far-approach", 150.0, 400.0, 0.15, 180.0, 0.70),
)


def stage_task(base: TaskConfig, stage: Stage) -> TaskConfig:
    """Give distant stages more time to close the gap."""
    steps = int(min(900, 260 + 1.3 * stage.range_max))
    return replace(base, max_steps=steps)
