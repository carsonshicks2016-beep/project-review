"""Curriculum stage configuration for BLADE.

A `StageConfig` fully specifies one rung of the curriculum: the world (terrain,
spawn distance), the episode rules (length, fall threshold, whether combat is
live), the reward weights, and the promotion criterion that graduates the agent
to the next rung.  `default_stages(loadout)` returns the standard ladder:

    STAND  -> learn to stand still & tall on flat ground
    WALK   -> learn to locomote toward the opponent on flat ground
    TERRAIN-> keep standing/moving on uneven ground
    SPAR   -> full armed combat on uneven ground
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageConfig:
    name: str
    loadout: str = "sword"
    terrain: bool = False
    bump: float = 0.0
    spawn_dist: float = 2.4
    max_seconds: float = 12.0
    fall_h: float = 0.72
    combat: bool = False
    # --- reward weights ---
    w_up: float = 0.5          # torso upright (up-vector .z)
    w_height: float = 0.5      # torso tall (toward standing)
    w_alive: float = 0.2       # per-step survival bonus
    w_crouch: float = 0.12     # slight knee flex (anti-lock)
    w_ctrl: float = 0.0008     # energy penalty
    w_forward: float = 0.0     # speed toward the opponent (locomotion)
    w_face: float = 0.0        # facing the opponent
    w_approach: float = 0.0    # closing the gap (delta distance)
    w_close: float = 0.0       # being in striking range
    w_far: float = 0.0         # penalty for loitering far
    w_tip: float = 0.0         # aiming the weapon at the opponent
    r_hit: float = 0.0
    r_hit_taken: float = 0.0
    r_block: float = 0.0
    r_fall: float = -3.0
    r_win: float = 0.0
    # --- promotion ---
    promote_metric: str = "survive"   # "survive" | "reach" | "none"
    promote_threshold: float = 0.85
    min_iters: int = 60               # don't promote before this many iters in-stage


def default_stages(loadout: str = "sword"):
    common = dict(loadout=loadout, fall_h=0.72)
    return [
        StageConfig(
            name="STAND", terrain=False, spawn_dist=6.0, max_seconds=60.0,
            w_up=0.6, w_height=0.6, w_alive=0.3, w_crouch=0.12, w_ctrl=0.001,
            promote_metric="survive", promote_threshold=0.86, min_iters=80, **common,
        ),
        StageConfig(
            name="WALK", terrain=False, spawn_dist=4.5, max_seconds=60.0,
            w_up=0.5, w_height=0.5, w_alive=0.25, w_forward=1.6, w_face=0.3, w_approach=1.5,
            w_crouch=0.1, w_ctrl=0.0008,
            promote_metric="reach", promote_threshold=0.7, min_iters=100, **common,
        ),
        StageConfig(
            name="TERRAIN", terrain=True, bump=0.6, spawn_dist=4.0, max_seconds=60.0,
            w_up=0.5, w_height=0.5, w_alive=0.25, w_forward=1.4, w_face=0.3, w_approach=1.5,
            w_crouch=0.1, w_ctrl=0.0008,
            promote_metric="reach", promote_threshold=0.6, min_iters=120, **common,
        ),
        StageConfig(
            name="SPAR", terrain=True, bump=0.85, spawn_dist=2.2, max_seconds=60.0,
            combat=True,
            w_up=0.45, w_height=0.45, w_alive=0.2, w_face=0.3, w_approach=2.5,
            w_close=0.5, w_far=0.12, w_tip=0.8, w_crouch=0.1, w_ctrl=0.0006,
            r_hit=3.5, r_hit_taken=1.2, r_block=0.4, r_win=5.0,
            promote_metric="none", min_iters=10 ** 9, **common,
        ),
    ]


STAGE_NAMES = [s.name for s in default_stages()]
