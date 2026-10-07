"""F3 — live agent stream: a replay that has not finished being written.

Runs a policy (or the reference pilot) in a dedicated env at wall-clock control
rate and streams **replay frames** over WebSocket. Pose/rendering uses the same
frame dicts as a saved replay; Observation debug for TRAIN G2 rides alongside
as ``sense`` on the live envelope (not on the frame — the replay schema forbids
unknown frame properties).

Fixed-rate stepping: one env step per control tick. If the tick is late we do
**not** double-step to catch up — we drop the missed wall time and resync.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from rallyai.control.frames import build_frame, build_sense, observation_to_sense
from rallyai.control.stage_resolve import resolve_stage
from rallyai.env import CONTROL_DT, EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.track import HIT_EDGE, HIT_OBSTACLE

CONTROL_HZ = 30
assert abs(CONTROL_DT - 1.0 / CONTROL_HZ) < 1e-9

LiveState = Literal["starting", "running", "stopping", "stopped", "error"]

PolicyFn = Callable[[np.ndarray, RallyEnv], np.ndarray]


class StartLiveBody(BaseModel):
    seed: int = Field(default=0)
    tier: int = Field(default=0, ge=0, le=32)
    checkpoint: str | None = Field(
        default=None,
        description="Optional policy checkpoint path (Phase C). Omit → reference pilot.",
    )
    procedural: bool = Field(default=True, description="Generate a stage from seed/tier.")
    stage: str | None = Field(
        default=None,
        description="Optional stage path or id; disables procedural when set.",
    )
    max_steps: int = Field(default=4500, ge=1, le=100_000)
    loop: bool = Field(
        default=False,
        description="If true, reset and continue after episode end until stopped.",
    )


@dataclass
class LiveRequest:
    seed: int = 0
    tier: int = 0
    checkpoint: str | None = None
    procedural: bool = True
    stage: str | None = None
    max_steps: int = 4500
    loop: bool = False


def _load_policy(checkpoint: str | None) -> tuple[PolicyFn, str, str | None]:
    """Return (act_fn, source_label, error). error set ⇒ caller should fail start."""
    if not checkpoint:

        def pilot_policy(obs: np.ndarray, env: RallyEnv) -> np.ndarray:
            del obs
            pilot = getattr(env, "_live_pilot", None)
            if pilot is None or getattr(pilot, "track", None) is not env.track:
                env._live_pilot = ReferencePilot(env.track)  # type: ignore[attr-defined]
                pilot = env._live_pilot  # type: ignore[attr-defined]
            return np.asarray(pilot.act(env.query, env.car), dtype=np.float32)

        return pilot_policy, "demo", None

    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        return (
            lambda *_: np.zeros(4, dtype=np.float32),
            "agent",
            f"checkpoint not found: {path}",
        )

    # Prefer the C4 eval loader (policy.mean_action + frozen normaliser).
    try:
        from rallyai.train.checkpoint import load_for_eval

        policy, normaliser, _meta = load_for_eval(path)

        def agent_policy(obs: np.ndarray, env: RallyEnv) -> np.ndarray:
            del env
            x = normaliser.normalize(np.asarray(obs, dtype=np.float32))
            return np.asarray(policy.mean_action(x), dtype=np.float32).reshape(-1)

        return agent_policy, "agent", None
    except Exception as exc:  # noqa: BLE001 — surface load errors to the API
        return (
            lambda *_: np.zeros(4, dtype=np.float32),
            "agent",
            f"failed to load checkpoint: {exc}",
        )


def _clip_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    out = np.zeros(4, dtype=np.float32)
    n = min(4, int(action.shape[0]))
    if n:
        out[:n] = action[:n]
    out[0] = float(np.clip(out[0], -1.0, 1.0))
    out[1] = float(np.clip(out[1], 0.0, 1.0))
    out[2] = float(np.clip(out[2], 0.0, 1.0))
    out[3] = float(np.clip(out[3], 0.0, 1.0))
    return out


class LiveSession:
    """One wall-clock env + frame buffer owned by the server."""

    def __init__(self, session_id: str, req: LiveRequest) -> None:
        self.session_id = session_id
        self.req = req
        self.state: LiveState = "starting"
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.source = "demo"
        self.checkpoint = req.checkpoint
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._packets: list[dict[str, Any]] = []
        self._header: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._steps = 0
        self._frame_i = 0
        self._termination: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "state": self.state,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "source": self.source,
                "checkpoint": self.checkpoint,
                "seed": self.req.seed,
                "tier": self.req.tier,
                "procedural": self.req.procedural,
                "stage": self.req.stage,
                "frames": len(self._packets),
                "steps": self._steps,
                "termination": self._termination,
                "dt": CONTROL_DT,
                "hz": CONTROL_HZ,
                "stream": f"/api/live/{self.session_id}/stream",
            }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"live-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
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
            }

    def _push(self, packet: dict[str, Any]) -> None:
        with self._cv:
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
        policy, source, err = _load_policy(self.req.checkpoint)
        self.source = source
        if err is not None:
            with self._lock:
                self.state = "error"
                self.error = err
                self.ended_at = time.time()
                self._cv.notify_all()
            return

        try:
            env = self._make_env()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.state = "error"
                self.error = f"failed to create env: {exc}"
                self.ended_at = time.time()
                self._cv.notify_all()
            return

        obs, _ = env.reset(seed=self.req.seed)
        env._live_pilot = ReferencePilot(env.track)  # type: ignore[attr-defined]

        header = {
            "type": "header",
            "session_id": self.session_id,
            "version": 1,
            "stage_id": str(env.stage.get("id", "")),
            "stage": env.stage,
            "dt": CONTROL_DT,
            "source": source,
            "meta": {
                "seed": self.req.seed,
                "tier": self.req.tier,
                "agent": source,
                "checkpoint": self.req.checkpoint,
                "physics_version": "supra@48908de",
                "car": env.car.spec.name,
            },
            "sense_channel": (
                "Each frame packet includes `frame` (replay schema) and `sense` "
                "(SensorSuite Observation debug: beam_*, lookahead_*, pace_note) "
                "for G2. beam_kinds use HIT_EDGE=0 / HIT_OBSTACLE=1. "
                "`frame` alone is enough for WATCH pose/rendering."
            ),
            "hit_kinds": {
                "HIT_EDGE": int(HIT_EDGE),
                "HIT_OBSTACLE": int(HIT_OBSTACLE),
            },
        }
        with self._lock:
            self._header = header
            self.state = "running"
            self.started_at = time.time()
            self._cv.notify_all()
        self._push(header)

        # Emit the reset frame so clients see t=0 pose immediately.
        self._emit_frame(env)
        self._steps = 0

        next_deadline = time.perf_counter() + CONTROL_DT
        episode_done = False

        while not self._stop.is_set():
            if episode_done:
                if not self.req.loop:
                    break
                obs, _ = env.reset(seed=self.req.seed)
                env._live_pilot = ReferencePilot(env.track)  # type: ignore[attr-defined]
                episode_done = False
                self._termination = None
                self._emit_frame(env)
                next_deadline = time.perf_counter() + CONTROL_DT
                continue

            action = _clip_action(policy(obs, env))
            obs, _reward, terminated, truncated, info = env.step(action)
            self._steps += 1
            self._emit_frame(env)

            if terminated or truncated:
                episode_done = True
                self._termination = str(info.get("termination") or "aborted")
                self._push(
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
                            "agent": source,
                            "termination": self._termination,
                        },
                        "frames": self._steps + 1,
                    }
                )

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
                next_deadline = time.perf_counter() + CONTROL_DT

        with self._lock:
            if self.state != "error":
                self.state = "stopped"
            self.ended_at = time.time()
            self._cv.notify_all()

    def _emit_frame(self, env: RallyEnv) -> None:
        frame = build_frame(env)
        if getattr(env, "_last_obs", None) is not None:
            sense = observation_to_sense(env._last_obs)
        else:
            sense = build_sense(env)
        idx = self._frame_i
        self._frame_i += 1
        self._push(
            {
                "type": "frame",
                "session_id": self.session_id,
                "i": idx,
                "frame": frame,
                "sense": sense,
            }
        )


class LiveStore:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.RLock()

    def start(self, req: LiveRequest) -> dict[str, Any]:
        session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        session = LiveSession(session_id, req)
        with self._lock:
            self._sessions[session_id] = session
        session.start()
        # Brief wait so checkpoint errors surface on the POST response.
        deadline = time.time() + 0.4
        while time.time() < deadline:
            snap = session.snapshot()
            if snap["state"] in ("running", "error", "stopped"):
                return snap
            time.sleep(0.02)
        return session.snapshot()

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        return session.stop()

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        return None if session is None else session.snapshot()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [s.snapshot() for s in sorted(sessions, key=lambda s: s.created_at, reverse=True)]

    def session(self, session_id: str) -> LiveSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def _get(self, session_id: str) -> LiveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session


def mount_live(app: FastAPI, store: LiveStore | None = None) -> LiveStore:
    """Register REST + WebSocket routes for live sessions."""
    live = store or LiveStore()
    app.state.live = live

    @app.post("/api/live", status_code=201)
    def start_live(body: StartLiveBody) -> dict[str, Any]:
        return live.start(
            LiveRequest(
                seed=body.seed,
                tier=body.tier,
                checkpoint=body.checkpoint,
                procedural=body.procedural,
                stage=body.stage,
                max_steps=body.max_steps,
                loop=body.loop,
            )
        )

    @app.get("/api/live")
    def list_live() -> dict[str, Any]:
        return {"sessions": live.list_sessions()}

    @app.get("/api/live/{session_id}")
    def get_live(session_id: str) -> dict[str, Any]:
        detail = live.get(session_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown live session: {session_id}")
        return detail

    @app.post("/api/live/{session_id}/stop")
    def stop_live(session_id: str) -> dict[str, Any]:
        try:
            return live.stop(session_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown live session: {session_id}"
            ) from exc

    @app.get("/api/live/{session_id}/frames")
    def get_frames(session_id: str, since: int = 0) -> dict[str, Any]:
        session = live.session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown live session: {session_id}")
        return session.read_since(since)

    @app.websocket("/api/live/{session_id}/stream")
    async def live_stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        session = live.session(session_id)
        if session is None:
            await websocket.send_json(
                {"type": "error", "message": f"unknown live session: {session_id}"}
            )
            await websocket.close(code=1008)
            return

        since_raw = websocket.query_params.get("since", "0")
        try:
            cursor = max(0, int(since_raw))
        except ValueError:
            cursor = 0

        try:
            await _stream_session(session, websocket, cursor)
        except WebSocketDisconnect:
            return
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except (OSError, RuntimeError):
                pass

    return live


async def _stream_session(
    session: LiveSession,
    websocket: WebSocket,
    since: int,
    poll_s: float = 0.01,
) -> None:
    cursor = since
    while True:
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
                        "message": session.error or "live session error",
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "stream_end",
                        "session_id": session.session_id,
                        "state": state,
                        "termination": batch.get("termination"),
                        "next": cursor,
                    }
                )
            return

        await asyncio.sleep(poll_s)
