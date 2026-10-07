from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BottleneckReport:
    label: str = "No major bottleneck yet"
    severity: float = 0.0
