"""Biome interface. A biome maps an Agent to a performance score + costs.

Each biome returns a BiomeResult with:
    score   primary performance (higher = better), in stated `unit`
    injury  injury-risk contribution in [0, ~2] (1.0 ≈ capacity limit)
    detail  sub-metrics for dashboards / explanation
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BiomeResult:
    name: str
    score: float
    unit: str
    injury: float
    detail: dict[str, float] = field(default_factory=dict)


class Biome:
    name: str = "base"
    unit: str = ""

    def evaluate(self, agent) -> BiomeResult:  # pragma: no cover - interface
        raise NotImplementedError
