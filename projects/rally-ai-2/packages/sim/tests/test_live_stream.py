"""F3 live agent stream: reference pilot, replay frames, sense sidecar."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rallyai.control.app import create_app
from rallyai.control.frames import build_frame, build_sense, observation_to_sense
from rallyai.control.stage_resolve import resolve_stage
from rallyai.env import EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.track import HIT_EDGE, HIT_OBSTACLE


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(runs_dir=tmp_path / "runs"))


def test_health_advertises_live(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert "live" in body
    assert "POST /api/live" in body["live"]["start"]


def test_build_frame_matches_replay_keys() -> None:
    stage = resolve_stage(seed=3, tier=0, procedural=True)
    env = RallyEnv(stage, EnvConfig(record_frames=True, max_time_s=2.0))
    env.reset(seed=3)
    pilot = ReferencePilot(env.track)
    env.step(pilot.act(env.query, env.car))
    frame = build_frame(env)
    assert {"t", "x", "y", "z", "yaw", "pitch", "roll", "v", "s"} <= set(frame)
    sense = build_sense(env)
    assert sense == observation_to_sense(env._last_obs)
    assert len(sense["beam_distances"]) == 9
    assert len(sense["lookahead_points"]) == 6
    assert set(sense["beam_kinds"]) <= {int(HIT_EDGE), int(HIT_OBSTACLE)}
    # Sense must stay off the frame (replay schema forbids unknown props).
    assert "sense" not in frame


def test_start_live_pilot_and_stream_frames(client: TestClient) -> None:
    r = client.post(
        "/api/live",
        json={"seed": 5, "tier": 0, "procedural": True, "max_steps": 200, "loop": False},
    )
    assert r.status_code == 201
    body = r.json()
    session_id = body["session_id"]
    assert body["source"] == "demo"
    assert body["state"] in ("starting", "running")
    assert body["stream"] == f"/api/live/{session_id}/stream"

    deadline = time.time() + 5.0
    packets: list[dict] = []
    while time.time() < deadline:
        page = client.get(f"/api/live/{session_id}/frames").json()
        packets = page["packets"]
        if any(p.get("type") == "frame" for p in packets):
            break
        time.sleep(0.05)
    assert any(p.get("type") == "header" for p in packets)
    header = next(p for p in packets if p.get("type") == "header")
    assert header["hit_kinds"]["HIT_OBSTACLE"] == int(HIT_OBSTACLE)
    assert header["hit_kinds"]["HIT_EDGE"] == int(HIT_EDGE)
    assert "stage" in header
    frames = [p for p in packets if p.get("type") == "frame"]
    assert len(frames) >= 1
    f0 = frames[0]
    assert "frame" in f0 and "sense" in f0
    assert {"t", "x", "y", "z", "yaw"} <= set(f0["frame"])
    assert "sense" not in f0["frame"]
    assert len(f0["sense"]["beam_distances"]) == 9

    with client.websocket_connect(f"/api/live/{session_id}/stream?since=0") as ws:
        msg = ws.receive_json()
        assert msg["type"] in ("header", "frame")
        if msg["type"] == "header":
            assert "stage" in msg
            msg = ws.receive_json()
        assert msg["type"] == "frame"
        for key in ("t", "x", "y", "z", "yaw"):
            assert key in msg["frame"]

    stop = client.post(f"/api/live/{session_id}/stop")
    assert stop.status_code == 200
    assert stop.json()["state"] in ("stopping", "stopped")

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if client.get(f"/api/live/{session_id}").json()["state"] == "stopped":
            break
        time.sleep(0.05)
    assert client.get(f"/api/live/{session_id}").json()["state"] == "stopped"


def test_bad_checkpoint_records_error(client: TestClient) -> None:
    r = client.post(
        "/api/live",
        json={"seed": 0, "tier": 0, "checkpoint": "/no/such/model.pt"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["state"] == "error"
    assert "checkpoint" in (body.get("error") or "").lower()


def test_unknown_live_404(client: TestClient) -> None:
    assert client.get("/api/live/does-not-exist").status_code == 404
    assert client.post("/api/live/does-not-exist/stop").status_code == 404
