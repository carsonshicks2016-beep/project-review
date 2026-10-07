"""C5 — metrics JSONL schema validation and flush behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rallyai.train.metrics import MetricsWriter, validate_metrics_line


def test_validate_run_start():
    validate_metrics_line(
        {
            "schema_version": 1,
            "kind": "run_start",
            "run_id": "x",
            "wall_t": 1.0,
            "msg": "hello",
        }
    )


def test_reject_unknown_field():
    with pytest.raises(Exception):
        validate_metrics_line(
            {
                "schema_version": 1,
                "kind": "update",
                "run_id": "x",
                "wall_t": 1.0,
                "not_a_field": True,
            }
        )


def test_writer_flush_every_line(tmp_path: Path):
    path = tmp_path / "smoke.jsonl"
    with MetricsWriter(path, "smoke") as w:
        w.write("run_start", msg="begin")
        w.write(
            "update",
            timesteps=100,
            tier=0,
            reward={"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "n": 1},
            terms={"progress": 0.5, "speed": 0.1},
            ppo={
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 1.0,
                "approx_kl": 0.01,
                "clip_frac": 0.1,
                "explained_var": 0.5,
                "lr": 3e-4,
                "grad_norm": 0.4,
                "rejected": False,
            },
        )
        w.write("run_end", timesteps=100, msg="done")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        validate_metrics_line(obj)
    assert json.loads(lines[0])["kind"] == "run_start"
    assert "terms" in json.loads(lines[1])
