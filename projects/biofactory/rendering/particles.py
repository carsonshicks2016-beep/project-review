from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Particle:
    x: float
    y: float
    life: float
    color: tuple[int, int, int]
