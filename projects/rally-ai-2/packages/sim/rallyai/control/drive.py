"""F4 — live human drive: keyboard inputs at control rate, replay frames out.

Browser sends ``[steer, throttle, brake, handbrake]``; the server owns a
``RallyEnv`` stepped at the **same 30 Hz control rate as training**. Network
jitter must not change physics: inputs are buffered and the loop steps **once
per tick**, holding the last input if none arrived. **Never** double-step to
catch up when a tick is late — drop the missed wall time and resync.

Outbound packets match F3 (``header`` / ``frame``+``sense`` / ``end`` /
``stream_end``) so WATCH and TRAIN overlays share one client path.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from rallyai import contracts
from rallyai.control.frames import build_frame, build_sense
from rallyai.control.stage_resolve import resolve_stage
from rallyai.env import CONTROL_DT, EnvConfig, RallyEnv
from rallyai.env.drive import stamp_human_meta

CONTROL_HZ = 30
ACTION_DIM = 4  # steer, throttle, brake, handbrake
assert abs(CONTROL_DT - 1.0 / CONTROL_HZ) < 1e-9

DriveState = Literal["starting", "running", "stopping", "stopped", "error"]


def default_human_baselines_dir() -> Path:
    """``docs/baselines/human/`` under the repo root."""
    # drive.py → control → rallyai → sim → packages → repo
    return Path(__file__).resolve().parents[4] / "docs" / "baselines" / "human"


def normalize_action(raw: Any) -> np.ndarray:
    """Coerce client payload to a clipped length-4 float32 action."""
    if isinstance(raw, dict):
        vals = [
            raw.get("steer", 0.0),
            raw.get("throttle", 0.0),
            raw.get("brake", 0.0),
            raw.get("handbrake", 0.0),
        ]
        arr = np.asarray(vals, dtype=np.float32)
    else:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    n = min(ACTION_DIM, int(arr.shape[0]))
    if n:
        out[:n] = arr[:n]
    out[0] = float(np.clip(out[0], -1.0, 1.0))
    out[1] = float(np.clip(out[1], 0.0, 1.0))
    out[2] = float(np.clip(out[2], 0.0, 1.0))
    out[3] = float(np.clip(out[3], 0.0, 1.0))
    return out


class StartDriveBody(BaseModel):
    seed: int = Field(default=0)
    tier: int = Field(default=0, ge=0, le=32)
    procedural: bool = Field(default=True)
    stage: str | None = Field(default=None)
    max_steps: int = Field(default=4500, ge=1, le=100_000)
    loop: bool = Field(
        default=True,
        description="If true, reset after episode end until the session is stopped.",
    )
    save_replay: bool = Field(
        default=True,
        description="Write a replay under docs/baselines/human/ when the session ends.",
    )
    replay_name: str | None = Field(
        default=None,
        description="Optional replay filename (``.json`` appended if missing).",
    )


@dataclass
class DriveRequest:
    seed: int = 0
    tier: int = 0
    procedural: bool = True
    stage: str | None = None
    max_steps: int = 4500
    loop: bool = True
    save_replay: bool = True
    replay_name: str | None = None


class DriveSession:
    """Server-owned human-drive env + fixed-rate stepper."""

    def __init__(
        self,
        session_id: str,
        req: DriveRequest,
        baselines_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.req = req
        self.baselines_dir = Path(baselines_dir) if baselines_dir else default_human_baselines_dir()
        self.state: DriveState = "starting"
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.replay_path: str | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._packets: list[dict[str, Any]] = []
        self._header: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._steps = 0
        self._frame_i = 0
        self._termination: str | None = None
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._env: RallyEnv | None = None
        self._episode_done = False
        # Exposed for unit tests — wall-clock loop increments this.
        self.ticks = 0
        self.catchup_drops = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "state": self.state,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "source": "human",
                "seed": self.req.seed,
                "tier": self.req.tier,
                "procedural": self.req.procedural,
                "stage": self.req.stage,
                "frames": len(self._packets),
                "steps": self._steps,
                "termination": self._termination,
                "dt": CONTROL_DT,
                "hz": CONTROL_HZ,
                "replay_path": self.replay_path,
                "stream": f"/api/drive/{self.session_id}/stream",
            }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"drive-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, save: bool | None = None) -> dict[str, Any]:
        if save is not None:
            self.req.save_replay = bool(save)
        self._stop.set()
        with self._lock:
            if self.state in ("starting", "running"):
                self.state = "stopping"
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self.state != "error":
                self.state = "stopped"
            if self.ended_at is None:
                self.ended_at = time.time()
            self._cv.notify_all()
        return self.snapshot()

    def set_action(self, action: Any) -> np.ndarray:
        """Buffer the latest client input (does not step physics)."""
        normalized = normalize_action(action)
        with self._lock:
            self._last_action = normalized.copy()
        return normalized

    def held_action(self) -> np.ndarray:
        with self._lock:
            return self._last_action.copy()

    def read_since(self, since: int) -> dict[str, Any]:
        with self._lock:
            since = max(0, int(since))
            batch = self._packets[since:]
            return {
                "session_id": self.session_id,
                "since": since,
                "next": since + len(batch),
                "header": self._header,
                "packets": list(batch),
                "state": self.state,
                "termination": self._termination,
                "replay_path": self.replay_path,
            }

    def tick_once(self) -> dict[str, Any] | None:
        """Advance the env exactly one control step (test + clock loop).

        Returns the frame packet, or ``None`` if the session cannot step
        (not ready / episode ended without loop / stopped).
        """
        with self._lock:
            if self.state not in ("running", "starting"):
                return None
            env = self._env
            if env is None:
                return None
            if self._episode_done:
                if not self.req.loop:
                    return None
                env.reset(seed=self.req.seed)
                self._episode_done = False
                self._termination = None
                return self._emit_frame_unlocked(env)

            action = self._last_action.copy()
            _obs, _reward, terminated, truncated, info = env.step(action)
            self._steps += 1
            self.ticks += 1
            packet = self._emit_frame_unlocked(env)
            if terminated or truncated:
                self._episode_done = True
                self._termination = str(info.get("termination") or "aborted")
                self._push_unlocked(
                    {
                        "type": "end",
                        "session_id": self.session_id,
                        "termination": self._termination,
                        "time_s": env.time_s,
                        "progress": float(info.get("progress", 0.0)),
                        "meta": {
                            "finished": bool(info.get("finished", False)),
                            "time_s": env.time_s,
                            "seed": self.req.seed,
                            "agent": "human",
                            "termination": self._termination,
                        },
                        "frames": self._steps + 1,
                    }
                )
            return packet

    def reset_episode(self) -> dict[str, Any] | None:
        with self._lock:
            env = self._env
            if env is None or self.state not in ("running", "starting"):
                return None
            env.reset(seed=self.req.seed)
            self._episode_done = False
            self._termination = None
            self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
            return self._emit_frame_unlocked(env)

    def save_replay_now(self, name: str | None = None) -> str | None:
        """Write the current env frames to the human baselines directory."""
        with self._lock:
            env = self._env
            if env is None:
                return None
            doc = stamp_human_meta(
                env.export_replay(source="human"),
                seed=self.req.seed,
                tier=self.req.tier,
                attempt=None,
            )
            if self._termination:
                doc["meta"]["termination"] = self._termination
            fname = name or self.req.replay_name
            if not fname:
                fname = f"human_{self.session_id}.json"
            if not fname.endswith(".json"):
                fname = f"{fname}.json"
            safe = Path(fname).name
            path = self.baselines_dir / safe

        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        try:
            contracts.write_json(doc, path, kind="replay")
        except Exception as exc:  # noqa: BLE001
            # Fall back to raw write if validation is unavailable / strict.
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self._lock:
                self.error = f"replay write warning: {exc}"
        with self._lock:
            self.replay_path = str(path)
            self._push_unlocked(
                {
                    "type": "saved",
                    "session_id": self.session_id,
                    "name": safe,
                    "path": str(path),
                }
            )
        return str(path)

    def _push(self, packet: dict[str, Any]) -> None:
        with self._cv:
            self._packets.append(packet)
            self._cv.notify_all()

    def _push_unlocked(self, packet: dict[str, Any]) -> None:
        self._packets.append(packet)
        self._cv.notify_all()

    def _make_env(self) -> RallyEnv:
        stage = resolve_stage(
            seed=self.req.seed,
            tier=self.req.tier,
            procedural=self.req.procedural,
            stage=self.req.stage,
        )
        max_time_s = float(self.req.max_steps) * CONTROL_DT
        return RallyEnv(
            stage,
            EnvConfig(record_frames=True, max_time_s=max_time_s),
        )

    def _run(self) -> None:
        try:
            env = self._make_env()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.state = "error"
                self.error = f"failed to create env: {exc}"
                self.ended_at = time.time()
                self._cv.notify_all()
            return

        env.reset(seed=self.req.seed)

        header = {
            "type": "header",
            "session_id": self.session_id,
            "version": 1,
            "stage_id": str(env.stage.get("id", "")),
            "stage": env.stage,
            "dt": CONTROL_DT,
            "source": "human",
            "meta": {
                "seed": self.req.seed,
                "tier": self.req.tier,
                "agent": "human",
                "physics_version": "supra@48908de",
                "car": env.car.spec.name,
            },
            "controls": {
                "rate_hz": CONTROL_HZ,
                "action": ["steer", "throttle", "brake", "handbrake"],
                "keys": "WASD / arrows; Space = handbrake; R = reset; P = save replay",
            },
            "sense_channel": (
                "Each frame packet includes `frame` (replay schema) and `sense` "
                "(Observation debug) — same envelope as F3 live agent stream."
            ),
        }
        with self._lock:
            self._env = env
            self._header = header
            self.state = "running"
            self.started_at = time.time()
            self._cv.notify_all()
        self._push(header)
        self._emit_frame(env)
        self._steps = 0

        next_deadline = time.perf_counter() + CONTROL_DT

        while not self._stop.is_set():
            if self._episode_done and not self.req.loop:
                break

            self.tick_once()

            # Fixed-rate: sleep until the next tick; never step twice to catch up.
            now = time.perf_counter()
            delay = next_deadline - now
            if delay > 0.0:
                end_wait = now + delay
                while not self._stop.is_set():
                    remaining = end_wait - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    time.sleep(min(remaining, 0.005))
            next_deadline += CONTROL_DT
            if next_deadline < time.perf_counter():
                # Drop missed ticks rather than bursting steps.
                self.catchup_drops += 1
                next_deadline = time.perf_counter() + CONTROL_DT

        if self.req.save_replay:
            try:
                self.save_replay_now()
            except OSError as exc:
                with self._lock:
                    self.error = f"replay save failed: {exc}"

        with self._lock:
            if self.state != "error":
                self.state = "stopped"
            self.ended_at = time.time()
            self._cv.notify_all()

    def _emit_frame(self, env: RallyEnv) -> dict[str, Any]:
        with self._cv:
            return self._emit_frame_unlocked(env)

    def _emit_frame_unlocked(self, env: RallyEnv) -> dict[str, Any]:
        frame = build_frame(env)
        sense = build_sense(env)
        idx = self._frame_i
        self._frame_i += 1
        packet = {
            "type": "frame",
            "session_id": self.session_id,
            "i": idx,
            "frame": frame,
            "sense": sense,
        }
        self._packets.append(packet)
        self._cv.notify_all()
        return packet


class DriveStore:
    def __init__(self, baselines_dir: Path | None = None) -> None:
        self.baselines_dir = Path(baselines_dir) if baselines_dir else default_human_baselines_dir()
        self._sessions: dict[str, DriveSession] = {}
        self._lock = threading.RLock()

    def start(self, req: DriveRequest) -> dict[str, Any]:
        session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        session = DriveSession(session_id, req, baselines_dir=self.baselines_dir)
        with self._lock:
            self._sessions[session_id] = session
        session.start()
        deadline = time.time() + 0.4
        while time.time() < deadline:
            snap = session.snapshot()
            if snap["state"] in ("running", "error", "stopped"):
                return snap
            time.sleep(0.02)
        return session.snapshot()

    def stop(self, session_id: str, *, save: bool | None = None) -> dict[str, Any]:
        return self._get(session_id).stop(save=save)

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        return None if session is None else session.snapshot()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [s.snapshot() for s in sorted(sessions, key=lambda s: s.created_at, reverse=True)]

    def session(self, session_id: str) -> DriveSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def _get(self, session_id: str) -> DriveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session


def mount_drive(
    app: FastAPI,
    store: DriveStore | None = None,
    baselines_dir: Path | None = None,
) -> DriveStore:
    """Register REST + WebSocket routes for human-drive sessions."""
    drive = store or DriveStore(baselines_dir=baselines_dir)
    app.state.drive = drive

    @app.post("/api/drive", status_code=201)
    def start_drive(body: StartDriveBody) -> dict[str, Any]:
        return drive.start(
            DriveRequest(
                seed=body.seed,
                tier=body.tier,
                procedural=body.procedural,
                stage=body.stage,
                max_steps=body.max_steps,
                loop=body.loop,
                save_replay=body.save_replay,
                replay_name=body.replay_name,
            )
        )

    @app.get("/api/drive")
    def list_drive() -> dict[str, Any]:
        return {"sessions": drive.list_sessions()}

    @app.get("/api/drive/{session_id}")
    def get_drive(session_id: str) -> dict[str, Any]:
        detail = drive.get(session_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown drive session: {session_id}")
        return detail

    @app.post("/api/drive/{session_id}/stop")
    def stop_drive(session_id: str, save: bool | None = None) -> dict[str, Any]:
        try:
            return drive.stop(session_id, save=save)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown drive session: {session_id}"
            ) from exc

    @app.post("/api/drive/{session_id}/input")
    def post_input(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """HTTP input path for tests / clients that cannot use the WS."""
        session = drive.session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown drive session: {session_id}")
        action = body.get("action", body)
        normalized = session.set_action(action)
        return {"ok": True, "action": normalized.tolist()}

    @app.get("/api/drive/{session_id}/frames")
    def get_frames(session_id: str, since: int = 0) -> dict[str, Any]:
        session = drive.session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown drive session: {session_id}")
        return session.read_since(since)

    @app.websocket("/api/drive/{session_id}/stream")
    async def drive_stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        session = drive.session(session_id)
        if session is None:
            await websocket.send_json(
                {"type": "error", "message": f"unknown drive session: {session_id}"}
            )
            await websocket.close(code=1008)
            return

        since_raw = websocket.query_params.get("since", "0")
        try:
            cursor = max(0, int(since_raw))
        except ValueError:
            cursor = 0

        try:
            await _stream_drive(session, websocket, cursor)
        except WebSocketDisconnect:
            return
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except (OSError, RuntimeError):
                pass

    return drive


async def _stream_drive(
    session: DriveSession,
    websocket: WebSocket,
    since: int,
    poll_s: float = 0.01,
) -> None:
    """Duplex: push frame packets; pull input / reset / save from the client."""
    cursor = since
    client_gone = asyncio.Event()

    async def recv_loop() -> None:
        try:
            await _recv_drive_messages(session, websocket)
        except WebSocketDisconnect:
            pass
        finally:
            client_gone.set()

    recv_task = asyncio.create_task(recv_loop())
    try:
        while not client_gone.is_set():
            batch = await asyncio.to_thread(session.read_since, cursor)
            for packet in batch["packets"]:
                await websocket.send_json(packet)
            cursor = int(batch["next"])

            state = batch["state"]
            if state in ("stopped", "error") and not batch["packets"]:
                if state == "error":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "session_id": session.session_id,
                            "message": session.error or "drive session error",
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "stream_end",
                            "session_id": session.session_id,
                            "state": state,
                            "termination": batch.get("termination"),
                            "replay_path": batch.get("replay_path"),
                            "next": cursor,
                        }
                    )
                return

            await asyncio.sleep(poll_s)
    finally:
        if not recv_task.done():
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass


async def _recv_drive_messages(session: DriveSession, websocket: WebSocket) -> None:
    while True:
        raw = await websocket.receive_json()
        if not isinstance(raw, dict):
            continue
        msg_type = raw.get("type", "input")
        if msg_type == "input":
            action = raw.get("action")
            if action is None:
                action = raw
            await asyncio.to_thread(session.set_action, action)
        elif msg_type == "reset":
            await asyncio.to_thread(session.reset_episode)
        elif msg_type == "save":
            name = raw.get("name")
            await asyncio.to_thread(session.save_replay_now, name)
        elif msg_type == "stop":
            await asyncio.to_thread(session.stop)
            return
