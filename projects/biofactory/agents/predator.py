from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Predator:
    predator_id: int
    x: float
    y: float
    health: float
    kind: str = "spider"
