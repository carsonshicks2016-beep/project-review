from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FitnessBreakdown:
    delivery: float = 0.0
    scouting: float = 0.0
    survival: float = 0.0
    congestion_penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.delivery + self.scouting + self.survival - self.congestion_penalty
