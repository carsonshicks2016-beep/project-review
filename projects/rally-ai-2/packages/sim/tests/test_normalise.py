"""Observation normaliser tests.

Load-bearing properties:
- freeze stops updates (eval order must not change the normaliser)
- save/load round-trips exactly
- a restored normaliser produces identical normalised observations
- clip is ±10
- no observation block is degenerate after a reference-pilot rollout
"""

from __future__ import annotations

import numpy as np
import pytest

from rallyai.env import EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.stage.generator import generate
from rallyai.train.normalise import (
    BLOCK_SLICES,
    OBS_CLIP,
    ObservationNormaliser,
)


def test_normalize_zero_init_is_identity_near_origin():
    n = ObservationNormaliser(4)
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    y = n.normalize(x)
    assert y.shape == (4,)
    assert np.allclose(y, 0.0)


def test_welford_matches_batch_mean_var():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(5000, 8)).astype(np.float64)
    n = ObservationNormaliser(8)
    # Stream in mini-batches the way rollout collection will.
    for i in range(0, len(data), 64):
        n.update(data[i:i + 64])
    assert n.count == pytest.approx(5000.0, abs=1.0)
    assert np.allclose(n.mean, data.mean(0), atol=1e-2)
    assert np.allclose(n.var, data.var(0), atol=5e-2)


def test_clip_is_plus_minus_ten():
    n = ObservationNormaliser(2, clip=OBS_CLIP)
    n.mean[:] = 0.0
    n.var[:] = 1e-12  # tiny variance → huge z-scores
    n.count = 1000.0
    out = n.normalize(np.array([100.0, -100.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx(OBS_CLIP)
    assert float(out[1]) == pytest.approx(-OBS_CLIP)


def test_freeze_blocks_updates():
    n = ObservationNormaliser(3)
    n.update(np.ones((10, 3)))
    mean_before = n.mean.copy()
    count_before = n.count
    n.freeze()
    assert n.update(np.full((10, 3), 99.0)) is False
    assert np.array_equal(n.mean, mean_before)
    assert n.count == count_before
    n.unfreeze()
    assert n.update(np.full((10, 3), 99.0)) is True
    assert not np.array_equal(n.mean, mean_before)


def test_state_dict_round_trip_exact():
    rng = np.random.default_rng(1)
    n = ObservationNormaliser(84)
    n.update(rng.normal(size=(200, 84)))
    n.update_returns(np.array([1200.0, 1800.0, 900.0]))
    payload = n.to_dict()

    restored = ObservationNormaliser.from_dict(payload)
    assert restored.dim == 84
    assert restored.count == n.count
    assert np.array_equal(restored.mean, n.mean)
    assert np.array_equal(restored.var, n.var)
    assert restored.ret_mean == n.ret_mean
    assert restored.ret_var == n.ret_var
    assert restored.ret_count == n.ret_count

    x = rng.normal(size=(84,)).astype(np.float32)
    assert np.array_equal(n.normalize(x), restored.normalize(x))


def test_load_into_existing_normaliser():
    rng = np.random.default_rng(2)
    a = ObservationNormaliser(16)
    a.update(rng.normal(size=(100, 16)))
    b = ObservationNormaliser(16)
    b.load_state_dict(a.state_dict())
    x = rng.normal(size=(16,)).astype(np.float32)
    assert np.array_equal(a.normalize(x), b.normalize(x))


def test_save_load_produces_identical_actions_proxy():
    """Roadmap Done when: restored normaliser → identical actions from identical obs.

    Without a policy yet, the proxy is: identical normalised observations, which
    are the policy inputs. Phase C wires the real action check through this.
    """
    rng = np.random.default_rng(3)
    n = ObservationNormaliser(84)
    n.update(rng.normal(size=(500, 84)))
    obs = rng.normal(size=(84,)).astype(np.float32)
    before = n.normalize(obs)

    twin = ObservationNormaliser.from_dict(n.to_dict())
    after = twin.normalize(obs)
    assert np.array_equal(before, after)


def test_rejects_wrong_dim():
    n = ObservationNormaliser(84)
    with pytest.raises(ValueError, match="obs dim"):
        n.update(np.zeros((4, 16), dtype=np.float32))


def test_drops_nonfinite_rows():
    n = ObservationNormaliser(2)
    bad = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]], dtype=np.float64)
    assert n.update(bad) is True
    # Only two finite rows contributed.
    assert n.count == pytest.approx(2.0, abs=1e-3)


def test_return_normalisation_tracks_scale():
    n = ObservationNormaliser(4)
    returns = np.array([800.0, 1200.0, 1600.0, 1000.0])
    n.update_returns(returns)
    # Prior count is 1e-4 (Welford seed), so the mean is within a hair of the sample.
    assert n.ret_mean == pytest.approx(float(returns.mean()), abs=0.05)
    z = n.normalize_returns(returns)
    assert abs(float(z.mean())) < 0.05


def test_recommended_vf_coef_shrinks_for_large_returns():
    n = ObservationNormaliser(4)
    n.update_returns(np.array([800.0, 1200.0, 1600.0, 1000.0]))
    note = n.return_scale_note()
    assert note["ret_std"] > 100.0
    # Stock 0.5 must be scaled down hard for thousand-scale returns.
    assert n.recommended_vf_coef() < 0.01
    assert note["recommended_vf_coef"] == pytest.approx(n.recommended_vf_coef())


def test_reference_pilot_blocks_not_degenerate():
    """B3 measure: after a pilot rollout, no obs *block* is dead.

    Sparse channels (obstacle-hit flags under a clean pilot; µ on a single-tier
    surface) can sit near the variance floor without the sensor being broken —
    the block still has live continuous channels. A truly dead block (every
    dim stuck) is the failure mode.
    """
    n = ObservationNormaliser(84)
    steps_target = 4_000  # enough signal; full 100k is a soak, not a unit test
    steps = 0
    seed = 40
    while steps < steps_target:
        # Rotate tiers so surface µ / width vary; otherwise lookahead_mu is
        # legitimately near-constant on homogeneous tier-0 gravel.
        tier = seed % 6
        stage = generate(seed, tier=tier)
        env = RallyEnv(stage, EnvConfig(max_time_s=25.0))
        pilot = ReferencePilot(env.track)
        obs, _ = env.reset(seed=seed)
        n.update(obs)
        steps += 1
        done = False
        while not done and steps < steps_target:
            action = pilot.act(env.query, env.car)
            obs, _r, term, trunc, _info = env.step(action)
            n.update(obs)
            steps += 1
            done = bool(term or trunc)
        seed += 1

    stats = n.block_stats()
    # Continuous sub-slices that must move if the suite is alive.
    continuous = {
        "vision_dist": slice(0, 9),       # ray distances (not binary hit flags)
        "proprio": BLOCK_SLICES["proprio"],
        "lookahead_curv": slice(43, 49),  # curvature at look-ahead times
        "terrain_core": slice(65, 72),    # grade/bank/pitch/roll/vz/height/air
    }
    for name, sl in continuous.items():
        var_min = float(n.var[sl].min())
        assert var_min > 1e-6, (name, var_min, n.var[sl])

    for name, sl in BLOCK_SLICES.items():
        assert name in stats
        # Block-level signal: mean variance across the block is not floor.
        assert stats[name]["var_mean"] > 1e-5, (name, stats[name])
        assert sl.stop - sl.start == {
            "vision": 18, "proprio": 25, "lookahead": 22, "terrain": 19,
        }[name]
