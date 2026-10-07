"""F1/F2 control room: job control + metrics JSONL (file is truth)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rallyai.control.app import create_app
from rallyai.control.jobs import TRAIN_MODULE, JobStore, RunRequest


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture()
def client(runs_dir: Path) -> TestClient:
    return TestClient(create_app(runs_dir=runs_dir))


def _fake_trainer_script(tmp_path: Path) -> Path:
    """Write a tiny stand-in for ``python -m rallyai.train.cli``."""
    script = tmp_path / "fake_train_cli.py"
    script.write_text(
        """\
import argparse, json, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run-id", required=True)
p.add_argument("--run-dir", required=True)
p.add_argument("--stage", required=True)
p.add_argument("--workers", type=int, required=True)
p.add_argument("--timesteps", type=int, required=True)
p.add_argument("--tier", type=int, required=True)
p.add_argument("--metrics", required=True)
p.add_argument("--abort", required=True)
args = p.parse_args()

metrics = Path(args.metrics)
abort = Path(args.abort)
wall0 = time.time()
with metrics.open("a") as f:
    f.write(json.dumps({
        "schema_version": 1,
        "kind": "run_start",
        "run_id": args.run_id,
        "wall_t": wall0,
        "timesteps": 0,
        "tier": args.tier,
        "msg": "fake trainer start",
    }) + "\\n")
    f.flush()
    step = 0
    deadline = time.time() + 8.0
    while time.time() < deadline and not abort.exists():
        step += 100
        f.write(json.dumps({
            "schema_version": 1,
            "kind": "update",
            "run_id": args.run_id,
            "wall_t": time.time(),
            "timesteps": step,
            "tier": args.tier,
            "reward": {"mean": float(step), "n": 1},
        }) + "\\n")
        f.flush()
        time.sleep(0.05)
    f.write(json.dumps({
        "schema_version": 1,
        "kind": "run_end",
        "run_id": args.run_id,
        "wall_t": time.time(),
        "timesteps": step,
        "msg": "fake trainer end",
    }) + "\\n")
"""
    )
    return script


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "metrics_polling" in body
    assert "live" in body
    assert "drive" in body


def test_list_empty(client: TestClient) -> None:
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert r.json()["runs"] == []


def test_unknown_run_404(client: TestClient) -> None:
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.get("/api/runs/does-not-exist/metrics").status_code == 404
    assert client.post("/api/runs/does-not-exist/stop").status_code == 404


def test_start_without_trainer_records_error_on_disk(
    client: TestClient, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawn failure still persists run dir + state.json (file is truth)."""
    import rallyai.control.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "TRAIN_MODULE", "rallyai.train._no_such_cli_module")
    r = client.post(
        "/api/runs",
        json={"stage": "proving_ground", "workers": 2, "timesteps": 1000, "tier": 1},
    )
    assert r.status_code == 201
    body = r.json()
    run_id = body["run_id"]
    assert body["state"] == "error"
    assert "rallyai.train._no_such_cli_module" in (body.get("error") or "")

    client2 = TestClient(create_app(runs_dir=runs_dir))
    detail = client2.get(f"/api/runs/{run_id}").json()
    assert detail["state"] == "error"
    assert (runs_dir / run_id / "state.json").exists()
    assert (runs_dir / run_id / "metrics.jsonl").exists()
    assert (runs_dir / run_id / "config.json").exists()

    listed = client2.get("/api/runs").json()["runs"]
    assert any(x["run_id"] == run_id for x in listed)


def test_metrics_replay_and_cursor(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_start(self: JobStore, req: RunRequest) -> dict:
        import uuid

        run_id = "test_" + uuid.uuid4().hex[:6]
        rd = self.run_dir(run_id)
        rd.mkdir()
        config = {
            "run_id": run_id,
            "stage": req.stage,
            "workers": req.workers,
            "timesteps": req.timesteps,
            "tier": req.tier,
            "train_module": TRAIN_MODULE,
        }
        (rd / "config.json").write_text(json.dumps(config))
        lines = [
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
                "timesteps": 100,
                "reward": {"mean": 1.0, "n": 1},
            },
            {
                "schema_version": 1,
                "kind": "run_end",
                "run_id": run_id,
                "wall_t": 3.0,
                "timesteps": 100,
            },
        ]
        with (rd / "metrics.jsonl").open("w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return self._write_state(
            run_id,
            state="stopped",
            created_at=time.time(),
            started_at=time.time(),
            ended_at=time.time(),
            config=config,
            metrics_path=str(rd / "metrics.jsonl"),
            error=None,
            exit_code=0,
            pid=None,
        )

    monkeypatch.setattr(JobStore, "start", fake_start)
    client = TestClient(create_app(runs_dir=runs_dir))
    started = client.post("/api/runs", json={"stage": "x", "workers": 1, "timesteps": 10, "tier": 0})
    run_id = started.json()["run_id"]

    all_lines = client.get(f"/api/runs/{run_id}/metrics").json()
    assert all_lines["since"] == 0
    assert len(all_lines["lines"]) == 3
    assert all_lines["next"] == 3
    assert all_lines["lines"][0]["kind"] == "run_start"

    page = client.get(f"/api/runs/{run_id}/metrics", params={"since": 1}).json()
    assert len(page["lines"]) == 2
    assert page["lines"][0]["kind"] == "update"
    assert page["next"] == 3


def test_live_run_stop_and_stream(
    runs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_trainer_script(tmp_path)
    import subprocess
    import sys
    import threading

    def start_with_fake(self: JobStore, req: RunRequest) -> dict:
        import os
        import uuid

        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        rd = self.run_dir(run_id)
        rd.mkdir(parents=True, exist_ok=False)
        config = {
            "run_id": run_id,
            "stage": req.stage,
            "workers": int(req.workers),
            "timesteps": int(req.timesteps),
            "tier": int(req.tier),
            "train_module": "fake",
        }
        self._write_json(self.config_path(run_id), config)
        metrics = self.metrics_path(run_id)
        metrics.touch()
        abort = self.abort_path(run_id)
        self._write_state(
            run_id,
            state="starting",
            created_at=time.time(),
            started_at=time.time(),
            config=config,
            metrics_path=str(metrics),
            error=None,
            exit_code=None,
            ended_at=None,
            pid=None,
        )
        log_path = rd / "trainer.log"
        log_f = log_path.open("w")
        proc = subprocess.Popen(
            [sys.executable, str(script), "--run-id", run_id, "--run-dir", str(rd),
             "--stage", req.stage, "--workers", str(req.workers),
             "--timesteps", str(req.timesteps), "--tier", str(req.tier),
             "--metrics", str(metrics), "--abort", str(abort)],
            cwd=str(rd),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        self.pid_path(run_id).write_text(str(proc.pid) + "\n")
        with self._lock:
            self._procs[run_id] = proc
        state = self._write_state(run_id, state="running", pid=proc.pid)
        threading.Thread(
            target=self._wait_proc,
            args=(run_id, proc, log_f),
            daemon=True,
        ).start()
        return state

    monkeypatch.setattr(JobStore, "start", start_with_fake)
    client = TestClient(create_app(runs_dir=runs_dir))
    started = client.post(
        "/api/runs",
        json={"stage": "proving_ground", "workers": 1, "timesteps": 500, "tier": 0},
    )
    assert started.status_code == 201
    run_id = started.json()["run_id"]
    assert started.json()["state"] == "running"

    deadline = time.time() + 3.0
    while time.time() < deadline:
        payload = client.get(f"/api/runs/{run_id}/metrics").json()
        if len(payload["lines"]) >= 2:
            break
        time.sleep(0.05)
    assert len(client.get(f"/api/runs/{run_id}/metrics").json()["lines"]) >= 2

    with client.websocket_connect(f"/api/runs/{run_id}/stream?since=1") as ws:
        msg = ws.receive_json()
        assert msg.get("kind") in ("update", "run_start", "run_end") or msg.get("type")

    stop = client.post(f"/api/runs/{run_id}/stop")
    assert stop.status_code == 200
    assert stop.json()["state"] in ("stopping", "stopped")

    deadline = time.time() + 4.0
    while time.time() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["state"] in ("stopped", "error"):
            break
        time.sleep(0.05)
    assert client.get(f"/api/runs/{run_id}").json()["state"] in ("stopped", "error")
    final = client.get(f"/api/runs/{run_id}/metrics").json()
    assert any(line.get("kind") == "run_end" for line in final["lines"])
