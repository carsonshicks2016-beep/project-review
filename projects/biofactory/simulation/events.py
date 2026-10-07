from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class SimEvent:
    tick: int
    message: str
    severity: str = "info"


class EventLog:
    def __init__(self, limit: int = 80) -> None:
        self._events: Deque[SimEvent] = deque(maxlen=limit)

    def add(self, tick: int, message: str, severity: str = "info") -> None:
        self._events.appendleft(SimEvent(tick=tick, message=message, severity=severity))

    def recent(self, count: int = 8) -> list[SimEvent]:
        return list(self._events)[:count]
