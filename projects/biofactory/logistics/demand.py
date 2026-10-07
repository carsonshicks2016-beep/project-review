from __future__ import annotations

from dataclasses import dataclass

from biofactory.resources.resource_types import ResourceType


BASE_RESOURCE_WEIGHTS: dict[ResourceType, float] = {
    ResourceType.LEAVES: 1.0,
    ResourceType.SEEDS: 0.62,
    ResourceType.WATER: 1.20,
    ResourceType.PROTEIN: 1.10,
    ResourceType.WOOD_FIBER: 0.42,
    ResourceType.MINERALS: 0.38,
    ResourceType.SOIL: 0.34,
    ResourceType.RESIN: 0.48,
    ResourceType.DEAD_INSECTS: 0.76,
    ResourceType.DEAD_ANTS: 0.52,
    ResourceType.WASTE: 0.45,
    ResourceType.FUNGUS_SUBSTRATE: 0.72,
    ResourceType.FUNGUS: 0.92,
    ResourceType.NUTRIENT_PASTE: 0.92,
    ResourceType.PROTEIN_PASTE: 0.82,
    ResourceType.LARVAE_FOOD: 0.95,
    ResourceType.REINFORCED_SOIL: 0.45,
    ResourceType.STRUCTURAL_RESIN: 0.45,
    ResourceType.COMPOST: 0.58,
    ResourceType.FERTILIZER: 0.52,
    ResourceType.SOLDIER_FEED: 0.68,
}

CRITICAL_FRACTIONS: dict[ResourceType, float] = {
    ResourceType.LEAVES: 0.20,
    ResourceType.SEEDS: 0.12,
    ResourceType.WATER: 0.25,
    ResourceType.PROTEIN: 0.20,
    ResourceType.WOOD_FIBER: 0.10,
    ResourceType.MINERALS: 0.10,
    ResourceType.SOIL: 0.10,
    ResourceType.RESIN: 0.12,
    ResourceType.DEAD_INSECTS: 0.14,
    ResourceType.DEAD_ANTS: 0.08,
    ResourceType.WASTE: 0.10,
}

CRITICAL_URGENCY_BOOST = 0.35
FOOD_TO_LEAF_PRIORITY_BOOST = 0.55


@dataclass
class DemandSignal:
    current: float
    desired: float

    @property
    def ratio(self) -> float:
        if self.desired <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.desired - self.current) / self.desired))


@dataclass(frozen=True)
class ResourceDemandState:
    resource_type: ResourceType
    stored: float
    desired: float
    shortage_ratio: float
    urgency: float
    saturation: float
    priority: float
    is_critical: bool
    is_saturated: bool


def calculate_resource_demand(
    resource_type: ResourceType,
    stored: float,
    desired: float,
    *,
    food_shortage_ratio: float = 0.0,
) -> ResourceDemandState:
    if desired <= 0.0:
        return ResourceDemandState(
            resource_type=resource_type,
            stored=stored,
            desired=desired,
            shortage_ratio=0.0,
            urgency=0.0,
            saturation=0.0,
            priority=0.0,
            is_critical=False,
            is_saturated=False,
        )

    shortage_ratio = _clamp01((desired - stored) / desired)
    saturation = max(0.0, (stored - desired) / desired)
    is_saturated = stored >= desired
    critical_fraction = CRITICAL_FRACTIONS.get(resource_type, 0.20)
    is_critical = stored <= desired * critical_fraction

    urgency = shortage_ratio**1.4
    if is_critical and shortage_ratio > 0.0:
        urgency += CRITICAL_URGENCY_BOOST
    if resource_type == ResourceType.LEAVES and food_shortage_ratio > 0.0:
        urgency += FOOD_TO_LEAF_PRIORITY_BOOST * _clamp01(food_shortage_ratio)

    saturation_penalty = 1.0 / (1.0 + saturation * 5.0)
    if is_saturated and shortage_ratio <= 0.0:
        urgency *= 0.08

    base_weight = BASE_RESOURCE_WEIGHTS.get(resource_type, 1.0)
    priority = max(0.0, base_weight * urgency * saturation_penalty)

    return ResourceDemandState(
        resource_type=resource_type,
        stored=stored,
        desired=desired,
        shortage_ratio=shortage_ratio,
        urgency=urgency,
        saturation=saturation,
        priority=priority,
        is_critical=is_critical,
        is_saturated=is_saturated,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
