from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Corpse:
    x: float
    y: float
    amount: float
    source: str
    decay: float = 0.001
