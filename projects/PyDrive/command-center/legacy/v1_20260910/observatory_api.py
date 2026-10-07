"""Command Center transport for the Fable Five 3D Brain Observatory.

Checkpoint selection and simulation remain server-side.  Browser input is
limited to playback controls and an audio listener pose; it cannot provide a
filesystem path, physics state, observation, or policy action.
"""
from __future__ import annotations

import atexit
from hashlib import sha256
import json
from pathlib import Path
import queue
import socket
import threading
import time
import uuid
from typing import Any, Mapping

from flask import current_app, jsonify, make_response, request, send_from_directory
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from supra.fable_editions import (
    CheckpointRejected,
    catalog_payload,
    get_edition,
    resolve_checkpoint,
)
from supra.observatory import (
    CheckpointCompatibilityError,
    ObservatoryError,
    ObservatorySessionManager,
    SessionClosedError,
    SessionLimitError,
    ValidatedCheckpoint,
)


PROTOCOL = "fable-observatory-v1"
BROWSER_COOKIE = "fable_observatory_browser"
# A direct `/observatory/` visit has no dashboard-selected edition. Make that
# supported launch deterministic instead of returning an opaque 422 from the
# session API. The resolved car/checkpoint remains visible in the identity rail
# and every dashboard launch still supplies its explicit edition.
DEFAULT_EDITION_ID = "787b"
SESSION_MANAGER = ObservatorySessionManager(
    max_sessions=4, per_browser=1, cleanup_seconds=120.0
)
_TELEMETRY_OWNERS: set[str] = set()
_OWNER_LOCK = threading.Lock()
_REAPER_STARTED = False


def _json_message(kind: str, **payload: Any) -> str:
    return json.dumps({"type": kind, **payload}, allow_nan=False,
                      separators=(",", ":"))


def _checkpoint_descriptor(root: Path, edition_id: str,
                           explicit: str | None = None) -> tuple[ValidatedCheckpoint, dict]:
    resolved = resolve_checkpoint(
        root,
        edition_id,
        explicit_selection=(None if explicit in (None, "", "active-best") else explicit),
    )
    descriptor = ValidatedCheckpoint.from_record(
        root,
        resolved.checkpoint,
        resolution_warnings=resolved.resolution_warnings,
    )
    return descriptor, resolved.as_dict()


def _browser_identity() -> tuple[str, bool]:
    supplied = (request.headers.get("X-Observatory-Browser")
                or request.cookies.get(BROWSER_COOKIE) or "").strip()
    if supplied and len(supplied) <= 128 and supplied.replace("-", "").isalnum():
        return supplied, False
    # A generated cookie is the stable one-browser owner token.  The request
    # fingerprint only ensures parallel first requests from one browser do not
    # race into multiple sessions before the cookie lands.
    fingerprint = "|".join((request.remote_addr or "local",
                            request.user_agent.string or "browser"))
    return f"{sha256(fingerprint.encode()).hexdigest()[:20]}-{uuid.uuid4().hex[:12]}", True


def _follow_resolver(root: Path, edition_id: str):
    def resolve(_current: ValidatedCheckpoint):
        descriptor, _ = _checkpoint_descriptor(root, edition_id)
        return descriptor
    return resolve


def _normalize_control(message: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(message)
    if payload.get("type") == "control":
        payload["type"] = payload.get("command")
    command = str(payload.get("type") or payload.get("command") or "").lower()
    if command == "speed" and "speed" not in payload and "value" in payload:
        payload["speed"] = payload["value"]
    if command == "scrub" and not any(
            key in payload for key in ("seconds_ago", "sequence", "sim_time")):
        payload["seconds_ago"] = payload.get("value", 0.0)
    if command in ("listener", "camera") and "listener" not in payload:
        value = payload.get("value")
        if isinstance(value, Mapping):
            payload["listener"] = dict(value)
    return payload


def _send_transport_result(outbox: queue.Queue, result: Mapping[str, Any] | None) -> None:
    if not result:
        return
    if result.get("type") == "frame" and isinstance(result.get("frame"), Mapping):
        outbox.put(_json_message("frame", **dict(result["frame"])))
    else:
        kind = str(result.get("type") or "event")
        outbox.put(_json_message(kind, **{k: v for k, v in result.items()
                                          if k != "type"}))


def _receive_controls(ws, session, outbox: queue.Queue, stop: threading.Event,
                      frame_ack: threading.Event) -> None:
    while not stop.is_set():
        try:
            raw = ws.receive(timeout=1.0)
        except TimeoutError:
            continue
        except Exception:
            stop.set()
            return
        if raw is None:
            # simple-websocket returns None when this timed receive expires;
            # it does not raise TimeoutError.  An idle viewer must therefore
            # keep its one authoritative telemetry stream instead of being
            # mistaken for a disconnected peer every second.
            if getattr(ws, "connected", False):
                continue
            stop.set()
            return
        try:
            if isinstance(raw, bytes):
                raise ValueError("telemetry controls must be JSON text")
            message = json.loads(raw)
            if not isinstance(message, Mapping):
                raise ValueError("control payload must be an object")
            message_type = str(message.get("type") or "").lower()
            if message_type == "frame_ack":
                frame_ack.set()
                continue
            if message_type == "disconnect":
                # Let the send loop stop before the browser begins its close
                # handshake.  This avoids a final telemetry frame racing the
                # peer close and being reported by Chromium as post-close data.
                stop.set()
                return
            result = session.handle_control(_normalize_control(message))
            _send_transport_result(outbox, result)
        except Exception as exc:
            outbox.put(_json_message("warning", code="control_rejected",
                                     message=str(exc)))


def _drain_session_events(session, ws) -> None:
    for event in session.drain_events():
        name = str(event.get("name") or "event")
        kind = "warning" if name == "warning" else "event"
        ws.send(_json_message(kind, event=name,
                              **{k: v for k, v in event.items() if k != "name"}))


def _audio_control(ws) -> str | None:
    """Poll the small audio transport control set without blocking cadence."""
    try:
        raw = ws.receive(timeout=0)
    except TimeoutError:
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return None
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        return None
    command = str(message.get("type") or "") if isinstance(message, Mapping) else ""
    return command if command in ("mute", "resume") else None


def _finish_websocket(ws) -> None:
    """Finish the upgraded socket before Werkzeug can write an HTTP tail.

    ``simple-websocket`` owns a reader thread.  A Flask route can otherwise
    return in the narrow interval between that thread observing a peer close
    and closing the raw upgraded socket, which makes Werkzeug's empty HTTP
    response look like a malformed WebSocket frame to Chromium.
    """
    try:
        if getattr(ws, "connected", False):
            ws.close(reason=1000, message="normal closure")
    except ConnectionClosed:
        pass

    reader = getattr(ws, "thread", None)
    if reader is None or reader is threading.current_thread():
        return
    reader.join(timeout=0.5)

    # Werkzeug keeps rfile/wfile makefiles referencing the upgraded socket.
    # Even after Simple-WebSocket's reader exits and calls sock.close(), those
    # references can keep the descriptor writable long enough for Werkzeug to
    # append an empty HTTP response.  An unconditional shutdown invalidates the
    # shared descriptor before the route wrapper returns.
    raw_socket = getattr(ws, "sock", None)
    if raw_socket is not None:
        try:
            raw_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    if reader.is_alive():
        reader.join(timeout=0.5)


def _run_reaper() -> None:
    while True:
        time.sleep(15.0)
        SESSION_MANAGER.reap_expired()


def _start_reaper() -> None:
    global _REAPER_STARTED
    if _REAPER_STARTED:
        return
    _REAPER_STARTED = True
    threading.Thread(target=_run_reaper, name="observatory-reaper",
                     daemon=True).start()


def register_observatory(app, project_root: str | Path) -> None:
    """Register static, REST, telemetry-WS, and audio-WS interfaces."""
    root = Path(project_root).resolve()
    # Remake client lives at observatory/dist; legacy/observatory is frozen rollback.
    dist = root / "observatory" / "dist"
    # Watch 2.5D is a sibling client (watch25d/dist) on the same protocol —
    # not a skin inside observatory/. Same session/WS APIs below.
    watch25d_dist = root / "watch25d" / "dist"
    sock = Sock(app)
    _start_reaper()

    @app.get("/observatory/")
    def observatory_index():
        if not (dist / "index.html").is_file():
            return jsonify({
                "error": "Observatory client has not been built",
                "command": "cd observatory && npm install && npm run validate",
            }), 503
        response = make_response(send_from_directory(dist, "index.html"))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/observatory/<path:asset>")
    def observatory_asset(asset: str):
        if not dist.is_dir():
            return jsonify({"error": "Observatory client has not been built"}), 503
        return send_from_directory(dist, asset)

    @app.get("/watch25d/")
    def watch25d_index():
        if not (watch25d_dist / "index.html").is_file():
            return jsonify({
                "error": "Watch 2.5D client has not been built",
                "command": "cd watch25d && npm install && npm run build",
            }), 503
        response = make_response(send_from_directory(watch25d_dist, "index.html"))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/watch25d/<path:asset>")
    def watch25d_asset(asset: str):
        if not watch25d_dist.is_dir():
            return jsonify({"error": "Watch 2.5D client has not been built"}), 503
        return send_from_directory(watch25d_dist, asset)


    @app.get("/api/observatory/catalog")
    def observatory_catalog():
        payload = catalog_payload(root)
        session_status = SESSION_MANAGER.status()
        payload["sessions"] = {
            key: session_status[key]
            for key in ("active", "maximum", "per_browser", "cleanup_seconds")
        }
        payload["truth_boundaries"] = {
            "physics": "authoritative Python Fable simulation",
            "browser": "render and audio-listener only",
            "road": "runtime-smoothed simulator track; constant width; zero bank",
            "visual_world": "deterministic non-colliding context; not surveyed",
            "porsche_919": "legacy Fable approximation; not faithful-v2 or evidence",
        }
        return jsonify(payload)

    @app.post("/api/observatory/sessions")
    def observatory_create_session():
        data = request.get_json(silent=True) or {}
        try:
            edition = get_edition(str(data.get("edition") or DEFAULT_EDITION_ID))
            checkpoint = str(data.get("checkpoint") or "active-best")
            mode = str(data.get("mode") or "replay")
            if mode not in ("replay", "follow-active-best"):
                raise ValueError("mode must be replay or follow-active-best")
            seed = int(data.get("seed", 7))
            if seed < 0 or seed > 2**31 - 1:
                raise ValueError("seed must be between 0 and 2147483647")
            audio = bool(data.get("audio", True))
            descriptor, resolved = _checkpoint_descriptor(
                root, edition.id, checkpoint
            )
            browser_id, set_cookie = _browser_identity()
            session_id, session = SESSION_MANAGER.create(
                browser_id,
                descriptor,
                mode=mode,
                seed=seed,
                audio=audio,
                follow_resolver=(_follow_resolver(root, edition.id)
                                 if mode == "follow-active-best" else None),
            )
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 400
        except (ValueError, CheckpointRejected, CheckpointCompatibilityError) as exc:
            return jsonify({"error": str(exc)}), 422
        except SessionLimitError as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"error": f"session creation failed: {exc}"}), 500

        response = make_response(jsonify({
            "protocol": PROTOCOL,
            "session_id": session_id,
            "resolved_checkpoint": resolved,
            "track_hash": session.track_hash,
            "telemetry_socket": f"/api/observatory/ws/{session_id}",
            "audio_socket": (f"/api/observatory/audio/{session_id}"
                             if session.audio is not None else None),
            "hello": session.hello(),
        }), 201)
        if set_cookie:
            response.set_cookie(BROWSER_COOKIE, browser_id, max_age=60 * 60 * 24,
                                httponly=True, samesite="Strict")
        return response

    @app.delete("/api/observatory/sessions/<session_id>")
    def observatory_delete_session(session_id: str):
        deleted = SESSION_MANAGER.delete(session_id)
        return jsonify({"ok": True, "released": deleted})

    @sock.route("/api/observatory/ws/<session_id>")
    def observatory_telemetry_socket(ws, session_id: str):
        with _OWNER_LOCK:
            if session_id in _TELEMETRY_OWNERS:
                ws.send(_json_message("error", code="telemetry_already_connected",
                                      message="This session already has a telemetry stream"))
                _finish_websocket(ws)
                return
            _TELEMETRY_OWNERS.add(session_id)
        try:
            session = SESSION_MANAGER.connect(session_id)
        except KeyError:
            with _OWNER_LOCK:
                _TELEMETRY_OWNERS.discard(session_id)
            ws.send(_json_message("error", code="unknown_session",
                                  message="Unknown Observatory session"))
            _finish_websocket(ws)
            return

        stop = threading.Event()
        frame_ack = threading.Event()
        frame_ack.set()
        outbox: queue.Queue[str] = queue.Queue(maxsize=64)
        receiver = threading.Thread(
            target=_receive_controls,
            args=(ws, session, outbox, stop, frame_ack),
            name=f"observatory-controls-{session_id[:8]}", daemon=True,
        )
        receiver.start()
        try:
            ws.send(_json_message("hello", **session.hello()))
            next_tick = time.monotonic()
            while not stop.is_set():
                while True:
                    try:
                        payload = outbox.get_nowait()
                    except queue.Empty:
                        break
                    if stop.is_set() or not getattr(ws, "connected", False):
                        break
                    ws.send(payload)
                if stop.is_set() or not getattr(ws, "connected", False):
                    break
                _drain_session_events(session, ws)
                if stop.is_set() or not getattr(ws, "connected", False):
                    break
                now = time.monotonic()
                if now >= next_tick and frame_ack.is_set():
                    frame = session.tick()
                    interval = session.wall_interval
                    next_tick += interval
                    finished_tick = time.monotonic()
                    # Keep the long-run policy cadence locked to the original
                    # deadline. Small tick/audio-lock delays are recovered by a
                    # slightly shorter following interval instead of becoming
                    # permanent playback drift. A delay beyond a full interval
                    # is overload, so rebase rather than emit a frame burst.
                    if next_tick < finished_tick - interval:
                        next_tick = finished_tick + interval
                    # ``session.tick`` can be expensive under software WebGL
                    # capture.  Re-check the peer after it returns so an
                    # already-observed close never receives one last frame.
                    if (frame is not None and not stop.is_set()
                            and getattr(ws, "connected", False)):
                        frame_ack.clear()
                        ws.send(_json_message("frame", **frame))
                stop.wait(min(0.02, max(0.001, next_tick - time.monotonic())))
        except (ConnectionClosed, BrokenPipeError, ConnectionResetError,
                SessionClosedError):
            pass
        except Exception as exc:
            current_app.logger.exception(
                "Observatory telemetry stream failed for %s", session_id,
            )
            try:
                ws.send(_json_message("error", code="telemetry_stream_failed",
                                      message=str(exc)))
            except Exception:
                pass
        finally:
            stop.set()
            SESSION_MANAGER.disconnect(session_id)
            with _OWNER_LOCK:
                _TELEMETRY_OWNERS.discard(session_id)
            _finish_websocket(ws)

    @sock.route("/api/observatory/audio/<session_id>")
    def observatory_audio_socket(ws, session_id: str):
        try:
            session = SESSION_MANAGER.connect(session_id)
        except KeyError:
            ws.send(_json_message("error", code="unknown_session",
                                  message="Unknown Observatory session"))
            _finish_websocket(ws)
            return
        try:
            if session.audio is None:
                ws.send(_json_message("warning", code="audio_disabled",
                                      message="Audio is disabled for this session"))
                return
            interval = session.audio.chunk_frames / session.audio.sample_rate
            deadline = time.monotonic()
            muted = False
            while True:
                command = _audio_control(ws)
                if command == "mute":
                    muted = True
                elif command == "resume":
                    muted = False
                    deadline = time.monotonic()
                if muted:
                    time.sleep(0.01)
                    deadline = time.monotonic()
                    continue
                ws.send(session.render_audio_chunk().wire_bytes())
                deadline += interval
                time.sleep(max(0.0, deadline - time.monotonic()))
        except ConnectionClosed:
            pass
        except Exception as exc:
            try:
                ws.send(_json_message("warning", code="audio_muted",
                                      message=str(exc)))
            except Exception:
                pass
        finally:
            SESSION_MANAGER.disconnect(session_id)
            _finish_websocket(ws)


atexit.register(SESSION_MANAGER.close_all)


__all__ = ["PROTOCOL", "SESSION_MANAGER", "register_observatory"]
