#!/usr/bin/env python3
"""End-to-end REST/WebSocket validation for the Observatory transport.

The validator starts a second Command Center on an ephemeral loopback port.  It
never discovers, signals, or reuses an existing dashboard process, and it only
deletes sessions created by this invocation.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "command-center" / "server.py"

try:
    import requests
except ImportError as exc:  # pragma: no cover - dependency gate
    raise SystemExit(f"MISSING dependency: requests ({exc})") from exc
try:
    import websocket
except ImportError as exc:  # pragma: no cover - dependency gate
    raise SystemExit(f"MISSING dependency: websocket-client ({exc})") from exc


AUDIO_HEADER = struct.Struct("<4sHHIIQd")


def gate(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""), flush=True)
    if not condition:
        raise AssertionError(name)


def ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def finite_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False

    def walk(item: Any) -> bool:
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, dict):
            return all(walk(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return all(walk(v) for v in item)
        return True

    return walk(value)


def wait_for_server(base_url: str, process: subprocess.Popen,
                    timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not contacted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Command Center exited during startup ({process.returncode})")
        try:
            response = requests.get(
                f"{base_url}/api/observatory/catalog", timeout=1.0
            )
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"Command Center did not become ready: {last_error}")


def websocket_url(base_url: str, path: str) -> str:
    if path.startswith("ws://") or path.startswith("wss://"):
        return path
    return "ws" + base_url[4:] + (path if path.startswith("/") else f"/{path}")


def connect_ws(url: str, timeout: float = 8.0):
    # Explicit no-proxy options keep loopback tests independent of shell proxy
    # configuration.  NO_PROXY is also set in the child environment below.
    return websocket.create_connection(
        url, timeout=timeout, http_proxy_host=None, http_proxy_port=None,
        suppress_origin=True,
    )


def receive_json_until(ws, predicate: Callable[[dict[str, Any]], bool],
                       *, timeout: float = 8.0,
                       label: str = "message") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    seen: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        ws.settimeout(max(0.1, deadline - time.monotonic()))
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            break
        if raw is None:
            break
        if isinstance(raw, bytes):
            continue
        message = json.loads(raw)
        if not isinstance(message, dict):
            continue
        seen.append(message)
        if predicate(message):
            return message, seen
        if message.get("type") == "error":
            raise AssertionError(f"transport error while waiting for {label}: {message}")
    raise AssertionError(f"timed out waiting for {label}; seen={seen[-8:]}")


def send_control(ws, command: str, **payload: Any) -> None:
    ws.send(json.dumps({"type": command, **payload}, separators=(",", ":")))


def session_post(base_url: str, browser: str, payload: dict[str, Any]):
    return requests.post(
        f"{base_url}/api/observatory/sessions",
        headers={"X-Observatory-Browser": browser},
        json=payload,
        timeout=30.0,
    )


def delete_session(base_url: str, session_id: str) -> requests.Response:
    return requests.delete(
        f"{base_url}/api/observatory/sessions/{session_id}", timeout=10.0
    )


def edition(catalog: dict[str, Any], edition_id: str) -> dict[str, Any]:
    for item in catalog.get("editions", []):
        if item.get("id") == edition_id:
            return item
    raise AssertionError(f"catalog has no {edition_id} edition")


def main() -> int:
    port = ephemeral_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update({
        "PORT": str(port),
        "PYTHONUNBUFFERED": "1",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    })
    created: list[str] = []
    telemetry_ws = None
    audio_ws = None
    log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        print(f"== isolated Command Center on {base_url} ==", flush=True)
        catalog = wait_for_server(base_url, process)
        gate("catalog is finite and carries truth boundaries",
             finite_json(catalog)
             and catalog.get("schema") == "fable-observatory-catalog-v1"
             and catalog.get("truth_boundaries", {}).get("browser")
             == "render and audio-listener only")
        mazda = edition(catalog, "787b")
        porsche = edition(catalog, "919")
        mazda_name = (mazda.get("resolved_checkpoint") or {}).get("name")
        gate("both edition catalogues resolve a brain",
             bool(mazda_name)
             and bool((porsche.get("resolved_checkpoint") or {}).get("name")))

        print("== fail-closed REST selection ==", flush=True)
        invalid = session_post(base_url, "transport-invalid", {
            "edition": "787b", "checkpoint": f"../{mazda_name}",
            "mode": "replay", "audio": False,
        })
        gate("path traversal is rejected", invalid.status_code == 422,
             f"HTTP {invalid.status_code}")
        absolute = session_post(base_url, "transport-absolute", {
            "edition": "787b", "checkpoint": str(ROOT / str(mazda_name)),
            "mode": "replay", "audio": False,
        })
        gate("absolute checkpoint paths are rejected", absolute.status_code == 422,
             f"HTTP {absolute.status_code}")
        cross_car = session_post(base_url, "transport-cross-car", {
            "edition": "919", "checkpoint": mazda_name,
            "mode": "replay", "audio": False,
        })
        gate("cross-car checkpoint selection is rejected",
             cross_car.status_code == 422, f"HTTP {cross_car.status_code}")

        print("== create and session-limit REST contract ==", flush=True)
        primary = session_post(base_url, "transport-primary", {
            "edition": "787b", "checkpoint": "active-best",
            "mode": "replay", "seed": 7, "audio": True,
        })
        gate("session POST succeeds", primary.status_code == 201,
             f"HTTP {primary.status_code}: {primary.text[:300]}")
        body = primary.json()
        primary_id = str(body.get("session_id") or "")
        if primary_id:
            created.append(primary_id)
        gate("session response exposes resolved identity, hash, and sockets",
             bool(primary_id) and len(str(body.get("track_hash") or "")) == 64
             and str(body.get("telemetry_socket") or "").endswith(primary_id)
             and str(body.get("audio_socket") or "").endswith(primary_id)
             and body.get("hello", {}).get("checkpoint", {}).get("car") == "mazda787b")

        same_browser = session_post(base_url, "transport-primary", {
            "edition": "787b", "checkpoint": "active-best",
            "mode": "replay", "audio": False,
        })
        gate("one-session-per-browser cap returns conflict",
             same_browser.status_code == 409, f"HTTP {same_browser.status_code}")

        # Fill the remaining three global slots with server-resolved sessions.
        for index in range(1, 4):
            response = session_post(base_url, f"transport-extra-{index}", {
                "edition": "919" if index == 3 else "787b",
                "checkpoint": "active-best", "mode": "replay",
                "audio": False,
            })
            gate(f"global slot {index + 1}/4 can be allocated",
                 response.status_code == 201,
                 f"HTTP {response.status_code}: {response.text[:200]}")
            created.append(response.json()["session_id"])
        overflow = session_post(base_url, "transport-overflow", {
            "edition": "787b", "checkpoint": "active-best",
            "mode": "replay", "audio": False,
        })
        gate("fifth global session returns conflict",
             overflow.status_code == 409, f"HTTP {overflow.status_code}")

        # Free the auxiliary slots before websocket work; only our IDs are touched.
        for session_id in created[1:]:
            response = delete_session(base_url, session_id)
            gate(f"auxiliary session {session_id[:8]} is released",
                 response.status_code == 200
                 and response.json().get("released") is True)
        del created[1:]

        print("== live telemetry WebSocket ==", flush=True)
        telemetry_ws = connect_ws(websocket_url(base_url, body["telemetry_socket"]))
        hello, _ = receive_json_until(
            telemetry_ws, lambda message: message.get("type") == "hello",
            label="telemetry hello",
        )
        gate("telemetry hello binds protocol/checkpoint/track",
             hello.get("protocol") == "fable-observatory-v1"
             and hello.get("checkpoint", {}).get("policy_sha256")
             == body.get("hello", {}).get("checkpoint", {}).get("policy_sha256")
             and hello.get("track", {}).get("hash") == body.get("track_hash"))

        frames: list[dict[str, Any]] = []
        while len(frames) < 4:
            frame, _ = receive_json_until(
                telemetry_ws, lambda message: message.get("type") == "frame",
                label=f"telemetry frame {len(frames) + 1}",
            )
            frames.append(frame)
            send_control(telemetry_ws, "frame_ack",
                         sequence=int(frame["sequence"]))
        sequences = [int(frame["sequence"]) for frame in frames]
        times = [float(frame["sim_time"]) for frame in frames]
        gate("frames are finite with monotonic sequence/time",
             all(finite_json(frame) for frame in frames)
             and all(b > a for a, b in zip(sequences, sequences[1:]))
             and all(b > a for a, b in zip(times, times[1:])))

        send_control(telemetry_ws, "pause")
        paused, _ = receive_json_until(
            telemetry_ws,
            lambda message: message.get("type") == "event"
            and (message.get("event") == "paused"
                 or message.get("name") == "paused"),
            label="pause acknowledgement",
        )
        gate("pause control is acknowledged", paused.get("type") == "event")

        send_control(telemetry_ws, "step")
        stepped, _ = receive_json_until(
            telemetry_ws, lambda message: message.get("type") == "frame",
            label="single-step frame",
        )
        send_control(telemetry_ws, "frame_ack",
                     sequence=int(stepped["sequence"]))
        gate("single-policy-tick step advances while paused",
             int(stepped["sequence"]) > sequences[-1]
             and stepped.get("playback", {}).get("paused") is True)

        send_control(telemetry_ws, "speed", speed=2.0)
        speed_event, _ = receive_json_until(
            telemetry_ws,
            lambda message: message.get("type") == "event"
            and message.get("event") == "speed_changed",
            label="speed acknowledgement",
        )
        gate("2x speed control is accepted", float(speed_event.get("speed")) == 2.0)

        send_control(telemetry_ws, "reset")
        reset_event, _ = receive_json_until(
            telemetry_ws,
            lambda message: message.get("type") == "event"
            and message.get("event") == "reset",
            label="reset acknowledgement",
        )
        gate("reset creates a new episode boundary",
             int(reset_event.get("episode", 0)) >= 1)

        send_control(telemetry_ws, "scrub", sequence=sequences[1])
        scrubbed, _ = receive_json_until(
            telemetry_ws, lambda message: message.get("type") == "frame",
            label="scrubbed frame",
        )
        send_control(telemetry_ws, "frame_ack",
                     sequence=int(scrubbed["sequence"]))
        gate("scrub returns buffered historical telemetry",
             int(scrubbed["sequence"]) == sequences[1]
             and scrubbed.get("playback", {}).get("scrubbing") is True)

        send_control(telemetry_ws, "resume")
        receive_json_until(
            telemetry_ws,
            lambda message: message.get("type") == "event"
            and message.get("event") == "resumed",
            label="resume acknowledgement",
        )
        resumed_frame, _ = receive_json_until(
            telemetry_ws, lambda message: message.get("type") == "frame",
            label="resumed live frame",
        )
        send_control(telemetry_ws, "frame_ack",
                     sequence=int(resumed_frame["sequence"]))
        gate("resume returns to the live edge",
             int(resumed_frame["sequence"]) > int(stepped["sequence"]))

        print("== live audio WebSocket ==", flush=True)
        audio_ws = connect_ws(websocket_url(base_url, body["audio_socket"]))
        audio_ws.settimeout(10.0)
        packet = audio_ws.recv()
        gate("audio websocket emits a binary packet", isinstance(packet, bytes))
        header = AUDIO_HEADER.unpack(packet[:AUDIO_HEADER.size])
        magic, version, channels, sample_rate, frame_count, audio_sequence, audio_time = header
        gate("FOA1 header declares 1024-frame 48k stereo s16",
             magic == b"FOA1" and version == 1 and channels == 2
             and sample_rate == 48_000 and frame_count == 1_024
             and len(packet) == AUDIO_HEADER.size + 1_024 * 2 * 2
             and audio_sequence >= 1 and math.isfinite(audio_time))

        audio_ws.send(json.dumps({"type": "mute"}))
        audio_ws.settimeout(0.35)
        muted = False
        # One chunk may already be in flight when the server observes the
        # control frame.  Once that bounded backlog drains, mute must leave
        # the socket alive without producing PCM.
        for _ in range(4):
            try:
                audio_ws.recv()
            except websocket.WebSocketTimeoutException:
                muted = True
                break
        gate("audio mute keeps the socket open and suppresses PCM", muted)

        audio_ws.send(json.dumps({"type": "resume"}))
        audio_ws.settimeout(10.0)
        resumed_packet = audio_ws.recv()
        gate("audio resume restarts binary PCM on the same socket",
             isinstance(resumed_packet, bytes)
             and resumed_packet[:4] == b"FOA1")

        audio_ws.close(status=1000, reason=b"acceptance teardown", timeout=3)
        gate("audio uses the standard WebSocket close handshake",
             not audio_ws.connected)
        audio_ws = None
        telemetry_ws.close()
        telemetry_ws = None

        print("== explicit DELETE cleanup ==", flush=True)
        deleted = delete_session(base_url, primary_id)
        gate("DELETE releases simulation/audio/buffer resources",
             deleted.status_code == 200 and deleted.json().get("released") is True)
        created.remove(primary_id)
        deleted_again = delete_session(base_url, primary_id)
        gate("DELETE is idempotent",
             deleted_again.status_code == 200
             and deleted_again.json().get("released") is False)

        unknown_ws = connect_ws(
            websocket_url(base_url, body["telemetry_socket"]), timeout=5.0
        )
        try:
            unknown, _ = receive_json_until(
                unknown_ws,
                lambda message: message.get("type") == "error"
                and message.get("code") == "unknown_session",
                label="deleted-session rejection",
            )
            gate("deleted session cannot be reconnected",
                 unknown.get("code") == "unknown_session")
        finally:
            unknown_ws.close()

        print("Observatory transport validation OK", flush=True)
        return 0
    except Exception:
        log.flush()
        log.seek(0)
        server_output = log.read()
        if server_output:
            print("\n--- isolated Command Center output (tail) ---", file=sys.stderr)
            print(server_output[-8000:], file=sys.stderr)
        raise
    finally:
        for ws in (audio_ws, telemetry_ws):
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        for session_id in reversed(created):
            try:
                delete_session(base_url, session_id)
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
