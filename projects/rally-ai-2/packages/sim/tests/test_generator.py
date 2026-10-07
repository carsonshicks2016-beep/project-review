"""Generator tests.

Two properties carry the most weight here:

* **Reproducibility.** Same version + seed + tier must give the same stage on any
  machine, or a replay is not a record and a held-out evaluation seed is not
  held out from anything.
* **The corridor invariant.** Nothing that can end the episode may sit inside the
  drivable corridor. The generator places obstacles through ``Track`` so this
  holds by construction, and it is checked here across many seeds and tiers
  because "by construction" is a claim, not a guarantee.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from rallyai import contracts
from rallyai.stage.generator import (
    N_TIERS,
    TIERS,
    crest_height_for_takeoff,
    generate,
    tier_params,
)
from rallyai.track import Track

# Kept small: every one of these builds a stage and a Track.
SEEDS = (0, 1, 7, 42, 1234)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_gives_an_identical_stage(seed):
    a = contracts.canonical_bytes(contracts.stamp(generate(seed, 2)))
    b = contracts.canonical_bytes(contracts.stamp(generate(seed, 2)))
    assert a == b


def test_different_seeds_give_different_stages():
    hashes = {contracts.content_hash(contracts.stamp(generate(s, 2))) for s in SEEDS}
    assert len(hashes) == len(SEEDS)


def test_tier_is_part_of_the_identity():
    a = contracts.content_hash(contracts.stamp(generate(5, 1)))
    b = contracts.content_hash(contracts.stamp(generate(5, 2)))
    assert a != b


def test_generation_uses_only_bit_stream_stable_rng():
    """NumPy guarantees the PCG64 bit stream across versions but not the output
    of distribution methods like ``normal()``. Using them would make stage
    identity depend on the numpy build, which is exactly the kind of silent
    cross-machine divergence the content hash exists to catch."""
    import inspect

    from rallyai.stage import generator

    src = inspect.getsource(generator)
    for banned in (".normal(", ".uniform(", ".choice(", ".standard_normal(",
                   ".exponential(", ".poisson("):
        assert banned not in src, f"{banned} is not guaranteed stable across numpy versions"


# --------------------------------------------------------------------------- #
# the corridor invariant
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tier", range(N_TIERS))
def test_no_obstacle_sits_inside_the_drivable_corridor(tier):
    """The hard rule from §5. v1 shipped a stage with 14 strikeable trees inside
    the corridor against an observation that could not see them."""
    for seed in (0, 3, 11):
        track = Track(generate(seed, tier))
        intruders = track.obstacles_in_corridor(car_radius=0.95)
        assert intruders == [], (
            f"tier {tier} seed {seed}: {len(intruders)} obstacles inside the corridor"
        )


@pytest.mark.parametrize("tier", range(N_TIERS))
def test_stages_carry_obstacles_at_all(tier):
    """Positive control for the test above: it would pass trivially on a stage
    with no obstacles."""
    stage = generate(0, tier)
    assert len(stage["obstacles"]) > 5


# --------------------------------------------------------------------------- #
# validity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tier", range(N_TIERS))
def test_generated_stage_validates_against_the_schema(tier):
    for seed in (0, 5):
        contracts.validate(contracts.stamp(generate(seed, tier)), "stage")


@pytest.mark.parametrize("tier", range(N_TIERS))
def test_generated_stage_builds_a_track(tier):
    for seed in (0, 5):
        track = Track(generate(seed, tier))
        assert track.length > 100.0
        assert np.all(np.isfinite(track.curvature))
        assert np.all(np.isfinite(track.z))
        assert np.all(track.width > 0)
        assert track.finish_s <= track.length


def test_tier_is_clamped_not_wrapped():
    assert tier_params(-5).name == TIERS[0].name
    assert tier_params(999).name == TIERS[-1].name


def test_stage_starts_and_finishes_on_something_straight():
    """A start line inside a corner is unfair and a finish line inside a hairpin
    is a lottery."""
    for seed in SEEDS:
        track = Track(generate(seed, 3))
        near_start = np.abs(track.curvature[:30])
        near_finish = np.abs(track.curvature[-30:])
        assert near_start.max() < 1 / 60.0
        assert near_finish.max() < 1 / 60.0


# --------------------------------------------------------------------------- #
# difficulty is one scalar
# --------------------------------------------------------------------------- #

def test_tier_parameters_move_together_monotonically():
    """§5: length, width, friction, corner severity, elevation aggression and
    obstacle proximity all move together. If one of them stops being monotonic
    the curriculum has more than one dial, and they can disagree."""
    for a, b in pairwise(TIERS):
        assert b.length_m[0] >= a.length_m[0], "stages must not get shorter"
        assert b.width_m[0] <= a.width_m[0], "corridors must not get wider"
        assert b.min_radius <= a.min_radius, "corners must not get easier"
        assert b.max_grade >= a.max_grade, "elevation must not get gentler"
        assert b.crest_chance >= a.crest_chance
        assert b.crest_takeoff_mps[0] <= a.crest_takeoff_mps[0], \
            "higher tiers must launch the car at LOWER speed"
        assert b.off_camber_chance >= a.off_camber_chance
        assert b.tree_clearance[0] <= a.tree_clearance[0], \
            "obstacles must not get further from the road"


def _surface_mu(name: str) -> float:
    from rallyai.stage.builder import DEFAULT_MU
    return DEFAULT_MU[name]


def test_worst_available_surface_never_improves():
    """The lowest grip a tier can throw at you is non-increasing.

    Note this is *not* the same as mean friction falling monotonically. Mean
    friction legitimately rises at tier 2, because that is where tarmac appears
    — and tarmac is in the vocabulary for the *transition* the agent has to read
    ahead for, not because it is slippery. Asserting on the mean would forbid
    ever introducing a high-grip surface.
    """
    worst = [min(_surface_mu(n) for n, _ in t.surfaces) for t in TIERS]
    for a, b in pairwise(worst):
        assert b <= a, f"the worst surface got better with tier: {worst}"


def test_the_hardest_tier_really_is_the_hardest_on_grip():
    """The specific regression this guards: tier 5's pool used to include tarmac
    at equal weight with snow, so the top tier drew high-grip surfaces as often
    as low-grip ones and measured EASIER than tier 4 — 6/12 completions against
    4/12. Pools are weighted for exactly this reason.
    """
    expected = [
        sum(w * _surface_mu(n) for n, w in t.surfaces) / sum(w for _, w in t.surfaces)
        for t in TIERS
    ]
    assert expected[-1] == min(expected), (
        f"the top tier is not the lowest-grip tier: {[round(e, 3) for e in expected]}"
    )
    assert expected[-1] < expected[-2], "grip must still be falling at the top"


@pytest.mark.parametrize("tier", range(N_TIERS))
def test_measured_geometry_respects_the_tier(tier):
    p = tier_params(tier)
    for seed in (0, 4, 9):
        stage = generate(seed, tier)
        track = Track(stage)
        # Radius floor, with the round-trip tolerance the fidelity tests measured.
        tightest = 1.0 / max(float(np.abs(track.curvature).max()), 1e-9)
        assert tightest > p.min_radius * 0.85, (
            f"tier {tier} seed {seed}: {tightest:.1f} m corner, floor is {p.min_radius}"
        )
        assert p.length_m[0] * 0.8 <= stage["length_m"] <= p.length_m[1] * 1.5


def test_elevation_stays_plausible():
    """A bounded random walk with no mean reversion drifts and stays drifted: it
    produced 190 m of climb over a 2 km stage, a sustained 10% gradient."""
    for tier in range(N_TIERS):
        for seed in range(6):
            track = Track(generate(seed, tier))
            span = float(track.z.max() - track.z.min())
            assert span < 0.12 * track.length, (
                f"tier {tier} seed {seed}: {span:.0f} m over {track.length:.0f} m"
            )


# --------------------------------------------------------------------------- #
# crests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("takeoff", [14.0, 20.0, 28.0])
@pytest.mark.parametrize("length", [26.0, 40.0])
def test_crest_launches_at_the_requested_speed(length, takeoff):
    """A crest is specified by the speed at which it launches the car, so that
    is what gets tested — not its height, which is meaningless on its own."""
    from rallyai.stage.builder import ProfileBuilder, build_stage

    h = crest_height_for_takeoff(length, takeoff)
    profile = ProfileBuilder(width=10.0).straight(60).crest(length, h).straight(60).build()
    track = Track(build_stage(profile, stage_id="c", name="c", seed=0, tier=0,
                              generator_version=1))
    # v = sqrt(g / |vcurv|) at the sharpest point
    measured = float(np.sqrt(9.81 / abs(track.vcurv.min())))
    assert measured == pytest.approx(takeoff, rel=0.15)


def test_low_tiers_do_not_launch_the_car():
    """Jumps are not an intro-tier problem. The tier 0 crest should need a speed
    the car cannot reach on a 400 m stage."""
    assert tier_params(0).crest_takeoff_mps[0] > 24.0
    assert tier_params(N_TIERS - 1).crest_takeoff_mps[1] < 24.0


# --------------------------------------------------------------------------- #
# sequencing
# --------------------------------------------------------------------------- #

def test_stages_use_a_vocabulary_not_a_random_walk():
    seen: set[str] = set()
    for seed in range(12):
        seen.update(generate(seed, 4)["meta"]["features"])
    # A road made only of straights and one corner type is a random walk with
    # extra steps.
    assert len(seen) >= 5, f"only {sorted(seen)} appeared across 12 stages"


def test_a_hairpin_is_preceded_by_somewhere_to_brake():
    """Sequencing with intent: a hairpin needs a straight in front of it."""
    from rallyai.stage.generator import TRANSITIONS

    for source, targets in TRANSITIONS.items():
        if source == "hairpin":
            assert targets.get("straight", 0) > 0.5, \
                "a hairpin must usually be followed by somewhere to accelerate"
    # and the hairpin archetype lays its own braking zone
    import inspect

    from rallyai.stage.generator import StageGenerator
    assert "b.straight(" in inspect.getsource(StageGenerator.hairpin)


def test_road_does_not_cross_itself():
    """A crossing would put one section's trees in another section's road."""
    from rallyai.stage.generator import _self_intersects

    for tier in range(N_TIERS):
        for seed in range(6):
            stage = generate(seed, tier)
            widest = max(p["width"] for p in stage["centerline"])
            assert not _self_intersects(stage, widest + 4.0), \
                f"tier {tier} seed {seed} folds over itself"
