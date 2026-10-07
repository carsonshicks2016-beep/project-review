from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from biofactory.resources.resource_types import ResourceType


@dataclass
class ColonyStats:
    ants_born: int = 0
    ants_died: int = 0
    leaves_collected: float = 0.0
    food_produced: float = 0.0
    food_consumed: float = 0.0
    water_consumed: float = 0.0
    protein_consumed: float = 0.0
    waste_generated: float = 0.0
    deliveries: int = 0
    distance_traveled: float = 0.0
    traffic_jam_events: int = 0
    resources_collected: dict[ResourceType, float] = field(default_factory=dict)
    recent_delivery_ticks: Deque[int] = field(default_factory=lambda: deque(maxlen=200))

    def record_delivery(self, tick: int, resource_type: ResourceType, amount: float) -> None:
        self.deliveries += 1
        self.resources_collected[resource_type] = self.resources_collected.get(resource_type, 0.0) + amount
        if resource_type == ResourceType.LEAVES:
            self.leaves_collected += amount
        self.recent_delivery_ticks.append(tick)

    def delivery_rate(self, tick: int, window: int = 600) -> float:
        if tick <= 0:
            return 0.0
        return sum(1 for t in self.recent_delivery_ticks if tick - t <= window) / max(1.0, window / 60.0)
