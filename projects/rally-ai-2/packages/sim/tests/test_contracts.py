"""The contracts are load-bearing, so they get tests before anything reads them.

These pin the three properties the rest of the project assumes:
  * the schemas are valid and cross-resolve,
  * a document's canonical form (and therefore its identity) is stable,
  * an edited file is detected rather than silently trusted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from rallyai import contracts

# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

SCHEMA_NAMES = ["stage", "replay", "metrics"]


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_valid(name):
    Draft202012Validator.check_schema(contracts.load_schema(name))


def test_replay_ref_to_stage_resolves():
    """A replay embeds a stage by $ref. If that reference stops resolving, replay
    validation silently stops checking the stage — the failure is invisible."""
    bad = _minimal_replay()
    bad["stage"] = {"schema_version": 1}  # missing everything else
    with pytest.raises(contracts.ContractError) as exc:
        contracts.validate(bad, "replay")
    # Proves the error came from inside stage.schema.json, not the replay schema.
    assert "generator" in str(exc.value)


# --------------------------------------------------------------------------- #
# canonical form and hashing
# --------------------------------------------------------------------------- #

def test_canonical_bytes_is_key_order_independent():
    a = {"b": 1, "a": {"d": 2, "c": 3}}
    b = {"a": {"c": 3, "d": 2}, "b": 1}
    assert contracts.canonical_bytes(a) == contracts.canonical_bytes(b)


def test_canonical_bytes_rejects_nan_and_inf():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            contracts.canonical_bytes({"x": bad})


def test_quantize_coerces_numpy_scalars_and_arrays():
    doc = {
        "f32": np.float32(1.2345678901),
        "f64": np.float64(2.5),
        "i64": np.int64(7),
        "arr": np.array([1.0, 2.0]),
        "nested": [{"v": np.float32(0.1)}],
    }
    out = contracts.quantize(doc)
    assert isinstance(out["f32"], float) and not isinstance(out["f32"], np.generic)
    assert isinstance(out["i64"], int)
    assert out["arr"] == [1.0, 2.0]
    assert isinstance(out["nested"][0]["v"], float)
    # json must accept the result — the whole point of quantising
    json.dumps(out)


def test_quantize_rounds_to_declared_precision():
    out = contracts.quantize({"x": 1.0 / 3.0})
    assert out["x"] == pytest.approx(0.333333, abs=1e-12)


def test_quantize_normalises_negative_zero():
    """Two runs differing only in the sign of a zero must hash identically."""
    assert contracts.canonical_bytes(contracts.quantize({"x": -0.0})) == \
           contracts.canonical_bytes(contracts.quantize({"x": 0.0}))


def test_content_hash_excludes_itself():
    doc = {"a": 1}
    h = contracts.content_hash(doc)
    stamped = dict(doc, content_hash=h)
    assert contracts.content_hash(stamped) == h


def test_stamp_is_idempotent():
    doc = {"a": 1.0 / 3.0, "b": [np.float32(2.0)]}
    once = contracts.stamp(doc)
    twice = contracts.stamp(once)
    assert once == twice


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #

def _minimal_stage() -> dict:
    return {
        "schema_version": 1,
        "id": "test_flat_01",
        "name": "Test Flat",
        "generator": {"version": 1, "seed": 0, "tier": 0, "authored": True},
        "length_m": 100.0,
        "centerline": [
            {"s": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "width": 8.0, "camber": 0.0},
            {"s": 100.0, "x": 100.0, "y": 0.0, "z": 0.0, "width": 8.0, "camber": 0.0},
        ],
        "surfaces": [{"s_start": 0.0, "s_end": 100.0, "type": "gravel", "mu": 0.72}],
        "pace_notes": [{"s": 0.0, "dir": "straight", "severity": 6, "text": "straight 100"}],
        "obstacles": [],
        "start": {"s": 0.0, "heading": 0.0},
        "finish": {"s": 100.0},
    }


def _minimal_replay() -> dict:
    return {
        "schema_version": 1,
        "stage": _minimal_stage(),
        "dt": 1.0 / 30.0,
        "source": "eval",
        "meta": {
            "termination": "finish",
            "time_s": 4.0,
            "car": "evo_rally",
            "physics_version": "48908de",
        },
        "frames": [
            {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0,
             "pitch": 0.0, "roll": 0.0, "v": 0.0, "s": 0.0},
        ],
    }


def test_minimal_stage_validates():
    contracts.validate(_minimal_stage(), "stage")


def test_minimal_replay_validates():
    contracts.validate(_minimal_replay(), "replay")


def test_replay_requires_exactly_one_stage_source():
    """oneOf: a replay carries the stage inline OR by reference, never both and
    never neither — otherwise 'which stage was this?' has two answers."""
    neither = _minimal_replay()
    del neither["stage"]
    with pytest.raises(contracts.ContractError):
        contracts.validate(neither, "replay")

    both = _minimal_replay()
    both["stage_ref"] = {"id": "test_flat_01", "content_hash": "sha256:" + "0" * 64}
    with pytest.raises(contracts.ContractError):
        contracts.validate(both, "replay")


def test_metrics_line_validates():
    line = {
        "schema_version": 1,
        "kind": "update",
        "run_id": "test",
        "wall_t": 1.0,
        "timesteps": 2048,
        "reward": {"mean": 12.0, "n": 8},
        "throughput": {"steps_per_s": 950.0, "n_workers": 1},
    }
    contracts.validate(line, "metrics")


def test_write_read_round_trip(tmp_path):
    path = contracts.write_json(_minimal_stage(), tmp_path / "s.json", kind="stage")
    back = contracts.read_json(path, kind="stage")
    assert back["id"] == "test_flat_01"
    assert back["content_hash"].startswith("sha256:")


def test_read_detects_edited_file(tmp_path):
    """The failure this prevents: a stage edited by hand, a replay recorded
    against the old geometry, and a car that appears to float."""
    path = contracts.write_json(_minimal_stage(), tmp_path / "s.json", kind="stage")
    doc = json.loads(path.read_text())
    doc["centerline"][1]["x"] = 999.0
    path.write_text(json.dumps(doc, indent=2))

    with pytest.raises(contracts.ContractError, match="content_hash mismatch"):
        contracts.read_json(path, kind="stage")


def test_written_file_is_human_readable(tmp_path):
    """§3: readable in a text editor and diffable in git. One centerline point
    per line, not a single 4000-character line."""
    path = contracts.write_json(_minimal_stage(), tmp_path / "s.json", kind="stage")
    text = path.read_text()
    assert "\n" in text
    assert max(len(line) for line in text.splitlines()) < 200


def test_write_rejects_invalid_document(tmp_path):
    bad = _minimal_stage()
    bad["surfaces"][0]["type"] = "lava"
    with pytest.raises(contracts.ContractError):
        contracts.write_json(bad, tmp_path / "bad.json", kind="stage")
