from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingMetrics:
    average_delivery_ticks: float = 0.0
    delivery_rate_per_10s: float = 0.0
    lost_cargo: float = 0.0
