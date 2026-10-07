from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


SCHEMA = "smw-bowser-ai-bridge-v1"


class ProtocolError(RuntimeError):
    """Raised when Lua/Python bridge messages are malformed."""


@dataclass(frozen=True)
class BridgeObservation:
    frame: int
    rom_hash: str = ""
    emulator: str = ""
    system_id: str = ""
    mode: str = "unknown"
    ram: dict[str, Any] = field(default_factory=dict)
    joypad: dict[str, bool] = field(default_factory=dict)
    screenshot_b64: str | None = None
    bridge_status: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeObservation":
        if payload.get("schema") not in {None, SCHEMA}:
            raise ProtocolError(f"unsupported observation schema: {payload.get('schema')}")
        frame = payload.get("frame", 0)
        if not isinstance(frame, int):
            raise ProtocolError("observation frame must be an int")
        ram = payload.get("ram", {})
        if not isinstance(ram, dict):
            raise ProtocolError("observation ram must be an object")
        joypad = payload.get("joypad", {})
        if not isinstance(joypad, dict):
            raise ProtocolError("observation joypad must be an object")
        return cls(
            frame=frame,
            rom_hash=str(payload.get("rom_hash", "")),
            emulator=str(payload.get("emulator", "")),
            system_id=str(payload.get("system_id", "")),
            mode=str(payload.get("mode", "unknown")),
            ram=ram,
            joypad={str(k): bool(v) for k, v in joypad.items()},
            screenshot_b64=payload.get("screenshot_b64"),
            bridge_status=payload.get("bridge", {}) if isinstance(payload.get("bridge", {}), dict) else {},
            raw=payload,
        )


@dataclass(frozen=True)
class BridgeAction:
    buttons: tuple[str, ...] = ()
    command: str = "step"
    frames: int = 1
    macro: str = "idle"
    evaluation_lock: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "command": self.command,
            "frames": max(1, int(self.frames)),
            "macro": self.macro,
            "buttons": list(self.buttons),
            "evaluation_lock": self.evaluation_lock,
            "note": self.note,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def make_length_prefixed(message: str) -> bytes:
    """BizHawk socket responses must be '<length> <message>'."""

    encoded = message.encode("utf-8")
    return f"{len(encoded)} ".encode("ascii") + encoded


def parse_length_prefixed(buffer: bytes) -> tuple[str, bytes]:
    space = buffer.find(b" ")
    if space < 1:
        raise ProtocolError("missing length prefix")
    length_bytes = buffer[:space]
    if not length_bytes.isdigit():
        raise ProtocolError("invalid length prefix")
    length = int(length_bytes)
    start = space + 1
    end = start + length
    if len(buffer) < end:
        raise ProtocolError("incomplete length-prefixed message")
    return buffer[start:end].decode("utf-8"), buffer[end:]


def extract_json_objects(text: str) -> tuple[list[dict[str, Any]], str]:
    """Extract complete top-level JSON objects from a streaming socket buffer."""

    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    last_end = 0

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                start = None
                depth = 0
                last_end = index + 1
                continue
            if depth == 0 and start is not None:
                raw = text[start : index + 1]
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ProtocolError(f"invalid JSON from Lua: {error}") from error
                if not isinstance(decoded, dict):
                    raise ProtocolError("Lua message must be a JSON object")
                objects.append(decoded)
                last_end = index + 1
                start = None

    return objects, text[last_end:]

