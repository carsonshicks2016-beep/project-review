"""Staged training pipeline: foundation → flow → fast → finish → frontier.

Each stage is a ``RewardConfig`` weighting shift over the same env and policy.
Lowering ``progress`` is the lever for a cautious driver, not raising ``speed``.
"""

from __future__ import annotations

from dataclasses import replace

from rallyai.env.reward import RewardConfig

STAGES = ("foundation", "flow", "fast", "finish", "frontier")


def stage_reward_config(stage: str, base: RewardConfig | None = None) -> RewardConfig:
    """Return the reward weighting for a named pipeline stage."""
    name = (stage or "foundation").strip().lower()
    if name not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    cfg = base or RewardConfig()

    if name == "foundation":
        # Survive and complete: progress dominant, pace terms near zero.
        return replace(
            cfg,
            progress=1.5,
            speed=0.5,
            speed_exp=1.0,
            throttle_commit=0.0,
            time_cost=0.15,
            smooth=0.2,
            finish=120.0,
            finish_pace=10.0,
            steer_jerk=0.04,
            steer_sat=0.05,
            overspeed=0.0,
        )
    if name == "flow":
        # Carry speed through the stage.
        return replace(
            cfg,
            progress=1.0,
            speed=18.0,
            speed_exp=2.0,
            throttle_commit=0.8,
            time_cost=0.4,
            smooth=0.4,
            finish=100.0,
            finish_pace=20.0,
            steer_jerk=0.06,
            steer_sat=0.10,
            overspeed=2.0,
        )
    if name == "fast":
        # Commit: throttle and time pressure up.
        return replace(
            cfg,
            progress=0.8,
            speed=22.0,
            speed_exp=2.2,
            throttle_commit=3.2,
            time_cost=0.9,
            smooth=0.3,
            finish=100.0,
            finish_pace=30.0,
            steer_jerk=0.08,
            steer_sat=0.15,
            overspeed=4.0,
        )
    if name == "finish":
        # Optimise the last seconds: finish_pace / time up, progress down.
        return replace(
            cfg,
            progress=0.4,
            speed=20.0,
            speed_exp=2.0,
            throttle_commit=2.4,
            time_cost=1.2,
            smooth=0.3,
            finish=80.0,
            finish_pace=50.0,
            steer_jerk=0.08,
            steer_sat=0.15,
            overspeed=5.0,
        )
    # frontier — full weights, intended for hardest tiers only.
    return replace(
        cfg,
        progress=1.0,
        speed=18.0,
        speed_exp=2.0,
        throttle_commit=2.4,
        time_cost=0.7,
        smooth=0.3,
        finish=100.0,
        finish_pace=35.0,
        steer_jerk=0.08,
        steer_sat=0.15,
        overspeed=4.0,
    )


def stage_tier_bounds(stage: str) -> tuple[int, int]:
    """``(min_tier, max_tier)`` hint for curriculum at each pipeline stage."""
    name = (stage or "foundation").strip().lower()
    if name == "foundation":
        return 0, 2
    if name == "flow":
        return 0, 3
    if name == "fast":
        return 1, 4
    if name == "finish":
        return 2, 5
    if name == "frontier":
        return 4, 5
    raise ValueError(f"unknown stage {stage!r}")
