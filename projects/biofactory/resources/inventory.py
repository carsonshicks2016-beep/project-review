from __future__ import annotations

from dataclasses import dataclass, field

from biofactory.resources.resource_types import ResourceType


@dataclass
class Inventory:
    amounts: dict[ResourceType, float] = field(default_factory=dict)

    def get(self, resource_type: ResourceType) -> float:
        return self.amounts.get(resource_type, 0.0)

    def add(self, resource_type: ResourceType, amount: float) -> None:
        if amount <= 0:
            return
        self.amounts[resource_type] = self.get(resource_type) + amount

    def remove(self, resource_type: ResourceType, amount: float) -> float:
        available = self.get(resource_type)
        removed = min(available, max(0.0, amount))
        remaining = available - removed
        if remaining <= 1e-9:
            self.amounts.pop(resource_type, None)
        else:
            self.amounts[resource_type] = remaining
        return removed

    def transfer_to(self, other: "Inventory", resource_type: ResourceType, amount: float) -> float:
        moved = self.remove(resource_type, amount)
        other.add(resource_type, moved)
        return moved
