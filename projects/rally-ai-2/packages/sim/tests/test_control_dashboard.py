"""Program dashboard APIs — disk-backed summary / checkpoints / evals / replays."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from rallyai.control.app import create_app


def _write_state(run_dir: Path, run_id: str, state: str = "running", **extra: object) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "state": state,
        "created_at": time.time(),
        "started_at": time.time(),
        "updated_at": time.time(),
        "config": {"run_id": run_id, "stage": "proving_ground", "workers": 1},
        "error": None,
        "exit_code": None,
        "ended_at": None,
        "pid": None,
    }
    payload.update(extra)
    (run_dir / "state.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_metrics(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in lines:
            f.write(json.dumps(row) + "\n")


def _write_ckpt(path: Path, *, timesteps: int, tier: int = 0, run_id: str = "r") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "timesteps": timesteps,
            "tier": tier,
            "run_id": run_id,
            "policy_sha256": "abc",
            "obs_dim": 4,
            "act_dim": 4,
        },
        path,
    )


def test_health_has_version_and_status(tmp_path: Path) -> None:
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "ok"
    assert body["version"]
    assert "dashboard" in body
    assert "replay_dirs" in body


def test_dashboard_summary_active_metrics_throughput_checkpoints(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_id = "dash_run_01"
    rd = runs / run_id
    # Keep a real PID alive so JobStore reconcile still reports "running".
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        _write_state(rd, run_id, state="running", pid=proc.pid)
        _write_metrics(
            rd / "metrics.jsonl",
            [
                {
                    "schema_version": 1,
                    "kind": "run_start",
                    "run_id": run_id,
                    "wall_t": 1.0,
                    "timesteps": 0,
                },
                {
                    "schema_version": 1,
                    "kind": "update",
                    "run_id": run_id,
                    "wall_t": 2.0,
                    "timesteps": 5000,
                    "tier": 1,
                    "throughput": {"steps_per_s": 1234.5, "n_workers": 4},
                    "reward": {"mean": 10.0, "n": 2},
                },
            ],
        )
        _write_ckpt(rd / f"{run_id}_latest.pt", timesteps=5000, tier=1, run_id=run_id)
        _write_ckpt(rd / f"{run_id}_best.pt", timesteps=4000, tier=1, run_id=run_id)

        # Flat CLI-style run also counted in checkpoint totals / discover.
        _write_ckpt(runs / "flat_latest.pt", timesteps=100, run_id="flat")
        _write_metrics(
            runs / "flat.jsonl",
            [
                {
                    "schema_version": 1,
                    "kind": "update",
                    "run_id": "flat",
                    "wall_t": 0.5,
                    "timesteps": 100,
                    "throughput": {"steps_per_s": 10.0, "n_workers": 1},
                }
            ],
        )

        client = TestClient(create_app(runs_dir=runs))
        r = client.get("/api/dashboard/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["active_count"] == 1
        assert body["active_runs"][0]["run_id"] == run_id
        assert body["latest_metrics"]["run_id"] == run_id
        assert body["latest_metrics"]["timesteps"] == 5000
        assert body["throughput"]["steps_per_s"] == 1234.5
        assert body["throughput"]["source"] == "metrics"
        assert body["checkpoints"]["total"] >= 3
        assert body["checkpoints"]["by_slot"]["latest"] >= 2
        assert body["checkpoints"]["by_slot"]["best"] >= 1
        assert body["runs_total"] >= 2
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_checkpoints_list_hof_with_mtime_and_timesteps(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_id = "hof_run"
    # Flat CLI layout (matches foundation_01_* on disk).
    _write_ckpt(runs / f"{run_id}_latest.pt", timesteps=12_000, tier=2, run_id=run_id)
    _write_ckpt(runs / f"{run_id}_best.pt", timesteps=11_000, tier=2, run_id=run_id)
    _write_ckpt(runs / f"{run_id}_cleanest.pt", timesteps=10_500, tier=1, run_id=run_id)
    (runs / f"{run_id}.jsonl").write_text("")

    client = TestClient(create_app(runs_dir=runs))
    assert client.get("/api/runs/missing/checkpoints").status_code == 404

    r = client.get(f"/api/runs/{run_id}/checkpoints")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    slots = {c["slot"]: c for c in body["checkpoints"] if c.get("slot")}
    assert set(slots) >= {"latest", "best", "cleanest"}
    assert slots["latest"]["timesteps"] == 12_000
    assert slots["latest"]["tier"] == 2
    assert slots["latest"]["mtime"] is not None
    assert slots["best"]["timesteps"] == 11_000


def test_evals_list_beside_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_id = "eval_run"
    rd = runs / run_id
    _write_state(rd, run_id, state="stopped")
    nested = {
        "kind": "eval",
        "tier": 0,
        "checkpoint": str(rd / f"{run_id}_latest.pt"),
        "completion_rate": 0.8,
        "clean_rate": 0.5,
    }
    (rd / f"{run_id}_latest.pt.tier0.eval.json").write_text(json.dumps(nested))
    flat = {
        "kind": "eval",
        "tier": 1,
        "completion_rate": 0.4,
        "eval": {"completion_rate": 0.4, "mean_time_s": 42.0},
    }
    (runs / f"{run_id}_best.pt.tier1.eval.json").write_text(json.dumps(flat))

    client = TestClient(create_app(runs_dir=runs))
    r = client.get(f"/api/runs/{run_id}/evals")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    assert len(body["evals"]) == 2
    tiers = {e.get("tier") for e in body["evals"]}
    assert tiers == {0, 1}
    assert any(e.get("completion_rate") == 0.8 for e in body["evals"])


def test_replays_list_viewer_public_and_runs_exports(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    viewer_replays = tmp_path / "viewer_replays"
    exports = runs / "exports"
    viewer_replays.mkdir()
    exports.mkdir()

    replay = {
        "schema_version": 1,
        "source": "agent",
        "frames": [{"t": 0.0}],
        "meta": {"seed": 7, "tier": 1},
    }
    (viewer_replays / "demo.json").write_text(json.dumps(replay))
    (exports / "export_a.json").write_text(
        json.dumps({**replay, "source": "eval", "meta": {"seed": 17, "tier": 0}})
    )
    # Non-replays must be ignored.
    (exports / "times.json").write_text(json.dumps({"times": []}))
    (exports / "noise.eval.json").write_text(json.dumps({"kind": "eval", "tier": 0}))

    client = TestClient(
        create_app(runs_dir=runs, replay_dirs=[viewer_replays, exports])
    )
    r = client.get("/api/replays")
    assert r.status_code == 200
    body = r.json()
    names = {x["name"] for x in body["replays"]}
    assert names == {"demo.json", "export_a.json"}
    assert all(x.get("frames") == 1 for x in body["replays"])
    sources = {x["source"] for x in body["replays"]}
    assert sources == {"agent", "eval"}


def test_summary_empty_runs_dir(tmp_path: Path) -> None:
    client = TestClient(create_app(runs_dir=tmp_path / "empty_runs"))
    r = client.get("/api/dashboard/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["active_count"] == 0
    assert body["active_runs"] == []
    assert body["checkpoints"]["total"] == 0
    assert body["latest_metrics"] is None
