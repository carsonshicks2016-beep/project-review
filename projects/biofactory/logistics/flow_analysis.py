from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowSnapshot:
    leaves_per_minute: float = 0.0
    food_per_minute: float = 0.0
