from __future__ import annotations

from dataclasses import dataclass

from biofactory.resources.resource_types import ResourceType


@dataclass
class StoragePolicy:
    resource_type: ResourceType
    desired: float
    capacity: float

    def fullness(self, current: float) -> float:
        return min(1.0, max(0.0, current / max(1.0, self.capacity)))

    def available_capacity(self, current: float) -> float:
        return max(0.0, self.capacity - current)

    def saturation(self, current: float) -> float:
        return max(0.0, (current - self.desired) / max(1.0, self.desired))

    def can_accept(self, current: float) -> bool:
        return self.available_capacity(current) > 1e-6
