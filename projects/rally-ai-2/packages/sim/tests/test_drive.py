"""Phase D3 offline human recorder — schema-valid replays, 30 Hz contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rallyai import contracts
from rallyai.env import CONTROL_DT
from rallyai.env.drive import (
    DEFAULT_ATTEMPTS_PER_SEED,
    HELD_OUT_MODULUS,
    HELD_OUT_REMAINDER,
    default_held_out_seeds,
    is_held_out_seed,
    main,
    make_env,
    build_stage,
    run_episode_scripted,
    scripted_action_fn,
    stamp_human_meta,
    write_human_replay,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TIMES_JSON = REPO_ROOT / "docs" / "baselines" / "human" / "times.json"
TIMES_SCHEMA = REPO_ROOT / "docs" / "baselines" / "human" / "times.schema.json"


def test_held_out_seed_convention():
    assert is_held_out_seed(7)
    assert is_held_out_seed(17)
    assert not is_held_out_seed(8)
    seeds = default_held_out_seeds(5)
    assert seeds == [7, 17, 27, 37, 47]
    assert all(s % HELD_OUT_MODULUS == HELD_OUT_REMAINDER for s in seeds)
    assert DEFAULT_ATTEMPTS_PER_SEED == 5


def test_scripted_drive_writes_valid_human_replay(tmp_path):
    stage = build_stage(7, 0, fixture=None)
    env = make_env(stage, max_time_s=30.0)
    run_episode_scripted(
        env,
        scripted_action_fn("throttle"),
        max_steps=45,
        seed=7,
    )
    path = write_human_replay(
        env,
        tmp_path / "human_seed7_tier0_a1.json",
        seed=7,
        tier=0,
        attempt=1,
    )
    doc = contracts.read_json(path, kind="replay")
    assert doc["source"] == "human"
    assert doc["dt"] == pytest.approx(CONTROL_DT, abs=1e-6)
    assert doc["meta"]["seed"] == 7
    assert doc["meta"]["tier"] == 0
    assert doc["meta"]["attempt"] == 1
    assert doc["meta"]["control_hz"] == pytest.approx(30.0, abs=1e-6)
    assert doc["meta"]["driver"] == "keyboard"
    assert len(doc["frames"]) >= 45


def test_cli_script_mode(tmp_path):
    out = tmp_path / "smoke.json"
    rc = main(
        [
            "--seed",
            "7",
            "--tier",
            "0",
            "--attempt",
            "1",
            "--script",
            "coast",
            "--max-steps",
            "30",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    doc = contracts.read_json(out, kind="replay")
    assert doc["source"] == "human"
    assert doc["dt"] == pytest.approx(CONTROL_DT, abs=1e-6)


def test_times_placeholder_validates_and_claims_nothing():
    import jsonschema

    times = json.loads(TIMES_JSON.read_text())
    schema = json.loads(TIMES_SCHEMA.read_text())
    jsonschema.validate(times, schema)
    assert times["seeds"] == []
    assert times["control_hz"] == 30
    assert "PLACEHOLDER" in times["notes"]
    assert times["attempts_per_seed"] == DEFAULT_ATTEMPTS_PER_SEED


def test_stamp_forces_human_source():
    fake = {
        "schema_version": 1,
        "dt": CONTROL_DT,
        "source": "agent",
        "meta": {"termination": "aborted", "time_s": 0.0, "car": "evo_rally",
                 "physics_version": "supra@48908de"},
        "frames": [{"t": 0, "x": 0, "y": 0, "z": 0, "yaw": 0, "pitch": 0, "roll": 0,
                    "v": 0, "s": 0}],
        "events": [],
        "stage": {"id": "x"},  # incomplete — stamp_human_meta only touches meta/source
    }
    out = stamp_human_meta(fake, seed=7, tier=0, attempt=2)
    assert out["source"] == "human"
    assert out["meta"]["seed"] == 7
    assert out["meta"]["attempt"] == 2
