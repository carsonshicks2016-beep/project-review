from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .protocol import (
    BridgeAction,
    BridgeObservation,
    ProtocolError,
    extract_json_objects,
    make_length_prefixed,
)


ObservationHandler = Callable[[BridgeObservation], BridgeAction]


@dataclass(frozen=True)
class BridgeStats:
    frames_seen: int
    connections: int
    last_frame: int | None
    last_message_at: float | None


class BridgeClient:
    """Python-side listener for BizHawk Lua frame messages."""

    def __init__(self, host: str = "127.0.0.1", port: int = 55355, *, timeout_s: float = 5.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._frames_seen = 0
        self._connections = 0
        self._last_frame: int | None = None
        self._last_message_at: float | None = None
        self._stop = threading.Event()

    def stats(self) -> BridgeStats:
        return BridgeStats(
            frames_seen=self._frames_seen,
            connections=self._connections,
            last_frame=self._last_frame,
            last_message_at=self._last_message_at,
        )

    def stop(self) -> None:
        self._stop.set()

    def serve_forever(self, handler: ObservationHandler) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    conn, _addr = server.accept()
                except TimeoutError:
                    continue
                except socket.timeout:
                    continue
                self._connections += 1
                with conn:
                    conn.settimeout(self.timeout_s)
                    self._serve_connection(conn, handler)

    def _serve_connection(self, conn: socket.socket, handler: ObservationHandler) -> None:
        text_buffer = ""
        while not self._stop.is_set():
            try:
                data = conn.recv(65536)
            except socket.timeout:
                continue
            if not data:
                return
            text_buffer += data.decode("utf-8", errors="replace")
            try:
                payloads, text_buffer = extract_json_objects(text_buffer)
            except ProtocolError as error:
                action = BridgeAction(note=f"protocol error: {error}")
                conn.sendall(make_length_prefixed(action.to_json()))
                text_buffer = ""
                continue
            for payload in payloads:
                try:
                    observation = BridgeObservation.from_dict(payload)
                    action = handler(observation)
                    self._frames_seen += 1
                    self._last_frame = observation.frame
                    self._last_message_at = time.time()
                except Exception as error:  # Keep emulator safe if Python policy crashes.
                    action = BridgeAction(note=f"handler error: {error}")
                conn.sendall(make_length_prefixed(action.to_json()))


def observation_to_pretty_json(observation: BridgeObservation) -> str:
    return json.dumps(observation.raw, indent=2, sort_keys=True)

