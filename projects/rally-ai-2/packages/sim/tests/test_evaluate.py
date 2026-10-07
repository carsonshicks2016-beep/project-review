"""Held-out evaluation harness (D1) and sector analysis (D4)."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rallyai.env import EnvConfig
from rallyai.stage.fixtures import proving_ground
from rallyai.stage.generator import generate
from rallyai.train.evaluate import (
    EVAL_SEED_MODULUS,
    EVAL_SEED_RESIDUE,
    ConstantMeanActor,
    FrozenIdentityNormaliser,
    ObsMeanActor,
    ReferencePilotActor,
    assert_train_eval_disjoint,
    evaluate,
    held_out_seeds,
    is_eval_seed,
    is_train_seed,
    load_actor_from_checkpoint,
    load_human_baseline,
    resolve_optimal_time,
    run_episode,
    sectors_for_stage,
)

# --------------------------------------------------------------------------- #
# Seed reservation
# --------------------------------------------------------------------------- #

def test_eval_residue_matches_roadmap_scheme():
    assert EVAL_SEED_MODULUS == 10
    assert EVAL_SEED_RESIDUE == 7
    assert is_eval_seed(7) and is_eval_seed(17) and is_eval_seed(107)
    assert is_train_seed(0) and is_train_seed(6) and is_train_seed(8)
    assert not is_train_seed(7)
    assert not is_eval_seed(8)


def test_held_out_seeds_are_reserved_and_disjoint_from_train_pool():
    eval_seeds = held_out_seeds(20)
    assert eval_seeds == [7 + 10 * i for i in range(20)]
    train_seeds = [s for s in range(200) if is_train_seed(s)]
    assert_train_eval_disjoint(train_seeds, eval_seeds)


def test_assert_train_eval_disjoint_catches_overlap():
    with pytest.raises(AssertionError, match="overlap"):
        assert_train_eval_disjoint([1, 7, 2], [7, 17])


def test_evaluate_rejects_non_held_out_seeds_by_default():
    with pytest.raises(ValueError, match="held-out"):
        evaluate(seeds=[0, 1], actor=ConstantMeanActor(), tier=0)


# --------------------------------------------------------------------------- #
# Traps: mean actions, frozen normaliser
# --------------------------------------------------------------------------- #

class _SamplingPolicy:
    """A policy that fails the test if anyone asks it to sample."""

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32)

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        raise AssertionError("eval must use mean actions, not sampling")


def test_obs_mean_actor_calls_mean_action_not_sample():
    actor = ObsMeanActor(_SamplingPolicy(), FrozenIdentityNormaliser())
    stage = proving_ground()
    from rallyai.env import RallyEnv

    env = RallyEnv(stage, EnvConfig(max_time_s=5.0))
    obs, _ = env.reset(seed=0)
    a = actor.action(env, obs)
    assert a.shape == (4,)


def test_frozen_normaliser_refuses_updates():
    norm = FrozenIdentityNormaliser()
    x = np.zeros(8, dtype=np.float32)
    assert np.array_equal(norm.normalize(x), x)
    with pytest.raises(RuntimeError, match="frozen"):
        norm.update(x)


def test_checkpoint_loader_rejects_corrupt(tmp_path: Path):
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(RuntimeError, match="failed to load checkpoint"):
        load_actor_from_checkpoint(ckpt)


def test_load_actor_from_checkpoint_finite_actions(tmp_path: Path):
    from rallyai.env.reward import RewardConfig
    from rallyai.sense import SensorSpec
    from rallyai.train.checkpoint import build_payload, save_checkpoint
    from rallyai.train.normalise import ObservationNormaliser
    from rallyai.train.policy import ActorCritic

    net = ActorCritic()
    norm = ObservationNormaliser(84)
    norm.update(np.random.randn(8, 84).astype(np.float32))
    path = tmp_path / "real.pt"
    save_checkpoint(
        path,
        build_payload(
            policy=net,
            optimizer=None,
            normaliser=norm,
            sensor_spec=SensorSpec(),
            reward_config=RewardConfig(),
            timesteps=0,
            tier=0,
        ),
    )
    actor, meta = load_actor_from_checkpoint(path)
    assert meta["has_normaliser"] is True
    obs = np.zeros(84, dtype=np.float32)
    action = actor.policy.mean_action(actor.normaliser.normalize(obs))
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))


# --------------------------------------------------------------------------- #
# Optimal-time stub interface
# --------------------------------------------------------------------------- #

def test_optimal_stub_accepts_injected_time():
    stage = proving_ground()
    assert resolve_optimal_time(stage, optimal_time_s=12.5) == 12.5


def test_optimal_injection_overrides_d2():
    stage = proving_ground()
    assert resolve_optimal_time(stage, optimal_time_s=9.0) == 9.0


def test_d2_optimal_is_used_when_present():
    pytest.importorskip("rallyai.stage.optimal")
    from rallyai.stage.optimal import theoretical_minimum_time

    stage = generate(7, 0)
    got = resolve_optimal_time(stage)
    assert got == pytest.approx(theoretical_minimum_time(stage), rel=1e-9)


# --------------------------------------------------------------------------- #
# Sectors (D4)
# --------------------------------------------------------------------------- #

def test_generated_stages_expose_archetype_sectors():
    stage = generate(7, 0)
    sectors = sectors_for_stage(stage)
    assert len(sectors) >= 3  # start + ≥1 feature + finish
    assert sectors[0]["name"] == "start"
    assert sectors[-1]["name"] == "finish"
    assert sectors[0]["s_start"] == 0.0
    # Contiguous and cover the stage length.
    for a, b in pairwise(sectors):
        assert a["s_end"] == pytest.approx(b["s_start"], abs=1e-6)
    assert sectors[-1]["s_end"] == pytest.approx(float(stage["length_m"]), abs=1e-3)
    assert "features" in stage["meta"]
    feature_names = [s["name"] for s in sectors[1:-1]]
    assert feature_names == stage["meta"]["features"]


def test_fixture_without_sectors_falls_back_to_whole_stage():
    stage = proving_ground()
    sectors = sectors_for_stage(stage)
    assert len(sectors) == 1
    assert sectors[0]["name"] == "whole"


# --------------------------------------------------------------------------- #
# End-to-end smoke on the reference pilot
# --------------------------------------------------------------------------- #

def test_evaluate_reference_pilot_writes_json_and_metrics(tmp_path: Path):
    metrics = evaluate(
        seeds=[7],
        tier=0,
        actor=ReferencePilotActor(),
        out_dir=tmp_path,
        optimal_time_s=15.0,
        env_config=EnvConfig(max_time_s=120.0),
    )
    path = Path(metrics["json_path"])
    assert path.exists()
    assert path.parent == tmp_path
    record = json.loads(path.read_text())
    assert record["seeds"] == [7]
    assert record["seed_reservation"]["eval_residue"] == 7
    assert metrics["json_path"] == str(path)
    assert 0.0 <= metrics["completion_rate"] <= 1.0
    assert 0.0 <= metrics["clean_rate"] <= 1.0
    assert set(metrics["terminations"]) >= {
        "finish", "crash", "off_course", "spun", "stuck", "timeout",
    }
    assert "mean_off_course_s" in metrics["cleanliness"]
    ep = record["episodes"][0]
    assert ep["optimal_time_s"] == 15.0
    assert ep["time_vs_optimal"] == pytest.approx(ep["time_s"] / 15.0)
    assert len(ep["sectors"]) >= 3
    # Sector times that were recorded should be non-negative.
    timed = [s["time_s"] for s in ep["sectors"] if s["time_s"] is not None]
    assert timed and all(t >= 0.0 for t in timed)


def test_evaluate_records_path_beside_checkpoint_stem(tmp_path: Path):
    ckpt = tmp_path / "hof_best.pt"
    ckpt.write_bytes(b"stub")
    # No C4 loader — supply an actor explicitly; path still keys off checkpoint.
    metrics = evaluate(
        seeds=[7],
        tier=0,
        checkpoint=ckpt,
        actor=ConstantMeanActor(),
        env_config=EnvConfig(max_time_s=2.0),  # intentional early timeout
    )
    assert metrics["json_path"].endswith("hof_best.pt.eval.json")
    assert Path(metrics["json_path"]).exists()


def test_run_episode_sector_times_sum_near_stage_time_on_finish():
    stage = generate(7, 0)
    ep = run_episode(
        stage,
        ReferencePilotActor(),
        seed=7,
        tier=0,
        env_config=EnvConfig(max_time_s=120.0),
        optimal_sector_times_s=None,
    )
    if not ep.finished:
        pytest.skip("reference pilot did not finish seed 7 tier 0")
    timed = [s["time_s"] for s in ep.sectors if s["time_s"] is not None]
    assert sum(timed) == pytest.approx(ep.time_s, rel=0.02, abs=0.15)


def test_injected_sector_optima_appear_in_report():
    stage = generate(7, 0)
    sectors = sectors_for_stage(stage)
    optima = [0.5 * (s["s_end"] - s["s_start"]) / 18.0 for s in sectors]
    ep = run_episode(
        stage,
        ConstantMeanActor(),
        seed=7,
        tier=0,
        env_config=EnvConfig(max_time_s=1.5),
        optimal_sector_times_s=optima,
    )
    assert [s["optimal_time_s"] for s in ep.sectors] == pytest.approx(optima)


def test_d2_sector_optima_fill_report_without_injection():
    stage = generate(7, 0)
    ep = run_episode(
        stage,
        ConstantMeanActor(),
        seed=7,
        tier=0,
        env_config=EnvConfig(max_time_s=1.5),
    )
    assert all(s["optimal_time_s"] is not None for s in ep.sectors)
    assert all(s["optimal_time_s"] > 0.0 for s in ep.sectors)


def test_human_baseline_empty_until_drives_exist():
    baseline = load_human_baseline()
    assert baseline == {}


def test_evaluate_scores_time_vs_human_when_baseline_present(tmp_path: Path):
    human = {
        "schema_version": 1,
        "control_hz": 30,
        "physics_version": "test",
        "car": "evo_rally",
        "held_out": {"modulus": 10, "remainder": 7},
        "attempts_per_seed": 1,
        "seeds": [
            {
                "seed": 7,
                "tier": 0,
                "best_time_s": 40.0,
                "median_time_s": 40.0,
                "attempts": [
                    {"attempt": 1, "time_s": 40.0, "replay": "x.json", "clean": True},
                ],
            }
        ],
    }
    path = tmp_path / "times.json"
    path.write_text(json.dumps(human), encoding="utf-8")
    metrics = evaluate(
        seeds=[7],
        tier=0,
        actor=ConstantMeanActor(),
        out_dir=tmp_path,
        human_baseline_path=path,
        env_config=EnvConfig(max_time_s=2.0),
    )
    # Constant actor will timeout; finished episodes only contribute to ratio.
    assert "time_vs_human" in metrics
    ep = metrics["episodes"][0]
    assert ep["human_best_s"] == 40.0
