from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Queen:
    x: float
    y: float
    health: float = 1.0
    egg_energy: float = 0.0
