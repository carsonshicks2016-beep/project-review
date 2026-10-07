"""Theoretical minimum stage time — forward-backward envelope.

A ratio of reference-pilot time to this bound below 1.0 means the bound is
wrong (a real drive cannot beat a lower bound). Measured across 24 seeds/tier
the finished-run means sit near 1.18–1.23× (see README); the soft upper assert
below only guards against an unrealistically tight bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from rallyai.env import EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.env.reward import RewardConfig
from rallyai.stage.fixtures import flat_straight, proving_ground
from rallyai.stage.generator import generate
from rallyai.stage.optimal import (
    theoretical_minimum_profile,
    theoretical_minimum_time,
)
from rallyai.track import Track


def test_flat_straight_is_bounded_by_traction_and_gearing():
    """On a long flat straight the bound is the time to accelerate to the
    gear-limited top speed and hold it — not the curvature formula, which is
    infinite on a straight."""
    stage = flat_straight(800.0, width=10.0)
    profile = theoretical_minimum_profile(stage)
    assert profile["time_s"] > 0.0
    assert profile["time_s"] < 800.0 / 20.0, "must clear 20 m/s average on gravel"
    assert profile["v"].max() <= profile["v_terminal"] + 1e-6
    # Standing start: the first sample is at the floor, not terminal speed.
    assert profile["v"][0] == pytest.approx(2.0, abs=0.05)


def test_tighter_corners_cut_average_speed():
    """A hairpin is a shorter arc than a sweeper at the same angle, so wall
    time alone is ambiguous — average speed is the fair comparison. The
    curvature cap must pull the hairpin's pace below the sweeper's."""
    from rallyai.stage.builder import ProfileBuilder, build_stage

    def _stage(radius: float) -> dict:
        b = ProfileBuilder(width=9.0, surface="gravel")
        b.straight(40)
        b.corner(radius, 160, "left", ease=6)
        b.straight(40)
        return build_stage(
            b.build(),
            stage_id=f"opt_r{int(radius)}",
            name=f"r{radius}",
            seed=0,
            tier=0,
            generator_version=1,
            authored=True,
        )

    sweeper = theoretical_minimum_profile(_stage(60.0))
    hairpin = theoretical_minimum_profile(_stage(12.0))
    assert hairpin["pace_ref_mps"] < sweeper["pace_ref_mps"] * 0.85, (
        f"hairpin {hairpin['pace_ref_mps']:.1f} m/s should be well below "
        f"sweeper {sweeper['pace_ref_mps']:.1f} m/s"
    )


def test_lower_mu_raises_the_bound():
    """Snow must be slower than gravel on identical geometry."""
    gravel = proving_ground()
    snow = {
        **gravel,
        "id": "proving_ground_snow",
        "surfaces": [
            {**seg, "mu": 0.36, "type": "snow"} for seg in gravel["surfaces"]
        ],
    }
    assert theoretical_minimum_time(snow) > theoretical_minimum_time(gravel) * 1.2


def test_camber_into_the_turn_lowers_the_bound():
    """Positive camber banks into a left-hander; that corner must be faster
    banked than flat, or the camber term is dead weight."""
    from rallyai.stage.builder import ProfileBuilder, build_stage

    def _left_hander(camber: float) -> dict:
        b = ProfileBuilder(width=9.0, surface="gravel")
        b.straight(50)
        b.corner(25, 120, "left", ease=8, camber=camber)
        b.straight(50)
        return build_stage(
            b.build(),
            stage_id="opt_camber",
            name="camber",
            seed=0,
            tier=0,
            generator_version=1,
            authored=True,
        )

    flat = theoretical_minimum_time(_left_hander(0.0))
    banked = theoretical_minimum_time(_left_hander(0.10))
    assert banked < flat, f"banked {banked:.3f}s should beat flat {flat:.3f}s"


def test_pure_function_does_not_mutate_the_stage():
    """The optimal time is derived. Writing it into the stage would violate
    ``generator.additionalProperties: false`` and the provenance contract."""
    stage = generate(201, 1)
    before = repr(stage)
    _ = theoretical_minimum_time(stage)
    assert repr(stage) == before
    assert "optimal_time" not in stage
    assert "optimal_time_s" not in stage.get("generator", {})


def test_env_holds_optimal_and_derives_pace_ref():
    """RallyEnv computes the bound once and rewrites pace_ref_mps from it."""
    stage = generate(201, 1)
    env = RallyEnv(stage, EnvConfig())
    assert env.optimal_time_s == pytest.approx(theoretical_minimum_time(stage),
                                               rel=1e-9)
    expected = env.track.finish_s / env.optimal_time_s
    assert env.reward_cfg.pace_ref_mps == pytest.approx(expected, rel=1e-9)
    # Caller-supplied RewardConfig is not mutated; the env holds a replace().
    cfg = RewardConfig(progress=2.0)
    env2 = RallyEnv(stage, reward_config=cfg)
    assert cfg.pace_ref_mps == 18.0
    assert env2.reward_cfg.progress == 2.0
    assert env2.reward_cfg.pace_ref_mps == pytest.approx(expected, rel=1e-9)


def test_bound_is_below_reference_pilot_on_proving_ground():
    """A real (if heuristic) drive must finish slower than the lower bound."""
    stage = proving_ground()
    theo = theoretical_minimum_time(stage)
    env = RallyEnv(stage, EnvConfig(record_frames=False))
    env.reset(seed=0)
    pilot = ReferencePilot(env.track)
    term = trunc = False
    steps = 0
    while not (term or trunc) and steps < 6000:
        _, _, term, trunc, info = env.step(pilot.act(env.query, env.car))
        steps += 1
    assert info["termination"] == "finish"
    assert info["time_s"] > theo
    ratio = info["time_s"] / theo
    assert 1.05 < ratio < 3.0, f"pilot/theo ratio {ratio:.2f} out of band"


@pytest.mark.parametrize("tier", [0, 1, 2, 3, 4, 5])
def test_pilot_to_optimal_ratio_band_across_tiers(tier):
    """Spot-check one seed per tier. Full 24-seed table lives in the README."""
    stage = generate(300 + tier, tier)
    theo = theoretical_minimum_time(stage)
    assert theo > 0.0
    env = RallyEnv(stage)
    env.reset(seed=0)
    pilot = ReferencePilot(env.track)
    term = trunc = False
    steps = 0
    while not (term or trunc) and steps < 8000:
        _, _, term, trunc, info = env.step(pilot.act(env.query, env.car))
        steps += 1
    if info["termination"] != "finish":
        pytest.skip(f"pilot did not finish tier {tier} seed {300 + tier}")
    ratio = info["time_s"] / theo
    assert ratio >= 1.0, f"bound beaten: ratio {ratio:.3f} on tier {tier}"
    assert ratio < 2.5, f"bound unrealistically tight: ratio {ratio:.3f}"


def test_profile_speed_never_exceeds_corner_cap():
    stage = generate(210, 2)
    profile = theoretical_minimum_profile(stage)
    # After both passes, speed must sit at or below the curvature cap.
    assert np.all(profile["v"] <= profile["v_corner"] + 1e-6)


def test_track_finish_matches_integrated_length():
    """pace_ref_mps = finish_s / time_s, so the integrated length must be the
    start→finish span the env uses for the finish bonus."""
    stage = generate(220, 2)
    profile = theoretical_minimum_profile(stage)
    track = Track(stage)
    assert profile["length_m"] == pytest.approx(track.finish_s - track.start_s,
                                                abs=1e-6)
    assert profile["pace_ref_mps"] == pytest.approx(
        profile["length_m"] / profile["time_s"], rel=1e-9
    )


def test_sector_optima_sum_to_stage_bound():
    from rallyai.stage.optimal import optimal_sector_times_s
    from rallyai.train.evaluate import sectors_for_stage

    stage = generate(7, 0)
    sectors = sectors_for_stage(stage)
    parts = optimal_sector_times_s(stage, sectors)
    assert len(parts) == len(sectors)
    assert all(t > 0.0 for t in parts)
    assert sum(parts) == pytest.approx(theoretical_minimum_time(stage), rel=1e-4)
