"""F4 live human drive: fixed-rate stepping, hold-last-input, replay export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rallyai.control.app import create_app
from rallyai.control.drive import (
    CONTROL_DT,
    CONTROL_HZ,
    DriveRequest,
    DriveSession,
    normalize_action,
)
from rallyai.control.stage_resolve import resolve_stage
from rallyai.env import EnvConfig, RallyEnv


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            runs_dir=tmp_path / "runs",
            baselines_dir=tmp_path / "baselines" / "human",
        )
    )


def _manual_session(
    tmp_path: Path,
    *,
    seed: int = 3,
    loop: bool = False,
    max_steps: int = 200,
) -> DriveSession:
    """Build a session ready for tick_once without starting the wall-clock thread."""
    session = DriveSession(
        "test_manual",
        DriveRequest(
            seed=seed,
            tier=0,
            procedural=True,
            max_steps=max_steps,
            loop=loop,
            save_replay=False,
        ),
        baselines_dir=tmp_path / "baselines" / "human",
    )
    stage = resolve_stage(seed=seed, tier=0, procedural=True)
    env = RallyEnv(
        stage,
        EnvConfig(record_frames=True, max_time_s=float(max_steps) * CONTROL_DT),
    )
    env.reset(seed=seed)
    session._env = env
    session.state = "running"
    session.started_at = time.time()
    return session


def test_normalize_action_clips_and_pads() -> None:
    a = normalize_action([2.0, -1.0, 0.5])
    assert a.shape == (4,)
    assert a[0] == 1.0
    assert a[1] == 0.0
    assert a[2] == 0.5
    assert a[3] == 0.0
    b = normalize_action({"steer": -0.5, "throttle": 1, "brake": 0, "handbrake": 1})
    assert np.allclose(b, [-0.5, 1.0, 0.0, 1.0])


def test_set_action_does_not_step_physics(tmp_path: Path) -> None:
    session = _manual_session(tmp_path)
    t0 = session._env.time_s  # type: ignore[union-attr]
    session.set_action([1.0, 1.0, 0.0, 0.0])
    session.set_action([-1.0, 0.0, 1.0, 1.0])
    assert session._env.time_s == t0  # type: ignore[union-attr]
    assert session._steps == 0
    assert session.ticks == 0


def test_hold_last_input_across_ticks(tmp_path: Path) -> None:
    session = _manual_session(tmp_path)
    session.set_action([0.25, 1.0, 0.0, 0.0])
    p1 = session.tick_once()
    assert p1 is not None and p1["type"] == "frame"
    assert session._steps == 1
    held = session.held_action()
    assert np.allclose(held, [0.25, 1.0, 0.0, 0.0])

    p2 = session.tick_once()
    assert p2 is not None
    assert session._steps == 2
    assert np.allclose(session.held_action(), held)
    assert session._env.car.speed > 0.0  # type: ignore[union-attr]


def test_tick_once_never_bursts(tmp_path: Path) -> None:
    session = _manual_session(tmp_path)
    session.set_action([0.0, 1.0, 0.0, 0.0])
    for i in range(5):
        session.tick_once()
        assert session._steps == i + 1
        assert abs(session._env.time_s - (i + 1) * CONTROL_DT) < 1e-9  # type: ignore[union-attr]


def test_wall_clock_steps_equal_ticks(tmp_path: Path) -> None:
    session = DriveSession(
        "wall_clock",
        DriveRequest(
            seed=3,
            tier=0,
            procedural=True,
            max_steps=80,
            loop=False,
            save_replay=False,
        ),
        baselines_dir=tmp_path / "baselines" / "human",
    )
    session.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if session.state == "running" and session.ticks >= 3:
            break
        time.sleep(0.02)
    session.stop(save=False)
    assert session.ticks >= 3
    assert session._steps == session.ticks
    assert session.state == "stopped"


def test_health_advertises_drive(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert "drive" in body
    assert "POST /api/drive" in body["drive"]["start"]
    assert body["drive"]["controls"]


def test_start_drive_stream_and_input(client: TestClient) -> None:
    r = client.post(
        "/api/drive",
        json={
            "seed": 5,
            "tier": 0,
            "procedural": True,
            "max_steps": 120,
            "loop": False,
            "save_replay": True,
            "replay_name": "test_human.json",
        },
    )
    assert r.status_code == 201
    body = r.json()
    session_id = body["session_id"]
    assert body["source"] == "human"
    assert body["hz"] == CONTROL_HZ
    assert body["stream"] == f"/api/drive/{session_id}/stream"

    deadline = time.time() + 5.0
    packets: list[dict] = []
    while time.time() < deadline:
        page = client.get(f"/api/drive/{session_id}/frames").json()
        packets = page["packets"]
        if any(p.get("type") == "frame" for p in packets):
            break
        time.sleep(0.05)
    assert any(p.get("type") == "header" for p in packets)
    frames = [p for p in packets if p.get("type") == "frame"]
    assert len(frames) >= 1
    assert "sense" in frames[0]
    assert "sense" not in frames[0]["frame"]
    assert len(frames[0]["sense"]["beam_distances"]) == 9

    inp = client.post(
        f"/api/drive/{session_id}/input",
        json={"steer": 0.1, "throttle": 1.0, "brake": 0.0, "handbrake": 0.0},
    )
    assert inp.status_code == 200
    assert inp.json()["action"][1] == 1.0

    with client.websocket_connect(f"/api/drive/{session_id}/stream?since=0") as ws:
        msg = ws.receive_json()
        assert msg["type"] in ("header", "frame", "saved")
        ws.send_json(
            {
                "type": "input",
                "steer": 0.0,
                "throttle": 1.0,
                "brake": 0.0,
                "handbrake": 0.0,
            }
        )

    stop = client.post(f"/api/drive/{session_id}/stop")
    assert stop.status_code == 200
    deadline = time.time() + 3.0
    detail = stop.json()
    while time.time() < deadline:
        detail = client.get(f"/api/drive/{session_id}").json()
        if detail["state"] == "stopped":
            break
        time.sleep(0.05)
    assert detail["state"] == "stopped"
    if detail.get("replay_path"):
        path = Path(detail["replay_path"])
        assert path.exists()
        doc = json.loads(path.read_text())
        assert doc["source"] == "human"
        assert len(doc["frames"]) >= 1


def test_live_agent_unaffected(client: TestClient) -> None:
    r = client.post(
        "/api/live",
        json={"seed": 2, "tier": 0, "procedural": True, "max_steps": 80, "loop": False},
    )
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["source"] == "demo"
    client.post(f"/api/live/{sid}/stop")


def test_unknown_drive_404(client: TestClient) -> None:
    assert client.get("/api/drive/does-not-exist").status_code == 404
    assert client.post("/api/drive/does-not-exist/stop").status_code == 404
