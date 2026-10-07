from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .protocol import BridgeAction, BridgeObservation


class JsonlTelemetryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlTelemetryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_frame(
        self,
        observation: BridgeObservation,
        action: BridgeAction,
        *,
        decision: str = "",
        reward: float | None = None,
        route_step: str | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "type": "frame",
            "frame": observation.frame,
            "mode": observation.mode,
            "ram": observation.ram,
            "action": action.to_dict(),
            "decision": decision,
        }
        if reward is not None:
            event["reward"] = reward
        if route_step is not None:
            event["route_step"] = route_step
        self.write_event(event)

    def write_event(self, event: dict[str, Any]) -> None:
        self._handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
        self._handle.flush()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def write_jsonl(path: str | Path, events: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")

