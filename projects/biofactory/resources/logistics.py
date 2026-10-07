from __future__ import annotations

from biofactory.pheromones.pheromone_config import PheromoneType
from biofactory.resources.resource_types import ResourceType


RAW_RESOURCE_TYPES: tuple[ResourceType, ...] = (
    ResourceType.LEAVES,
    ResourceType.SEEDS,
    ResourceType.WATER,
    ResourceType.PROTEIN,
    ResourceType.WOOD_FIBER,
    ResourceType.MINERALS,
    ResourceType.SOIL,
    ResourceType.RESIN,
    ResourceType.DEAD_INSECTS,
    ResourceType.DEAD_ANTS,
    ResourceType.WASTE,
)

PHASE2_PROCESSED_RESOURCES: tuple[ResourceType, ...] = (
    ResourceType.FUNGUS_SUBSTRATE,
    ResourceType.FUNGUS,
    ResourceType.NUTRIENT_PASTE,
    ResourceType.PROTEIN_PASTE,
    ResourceType.LARVAE_FOOD,
    ResourceType.REINFORCED_SOIL,
    ResourceType.STRUCTURAL_RESIN,
    ResourceType.COMPOST,
    ResourceType.FERTILIZER,
    ResourceType.SOLDIER_FEED,
)

PROCESSED_LOGISTICS_RESOURCES: tuple[ResourceType, ...] = (
    ResourceType.FOOD,
    *PHASE2_PROCESSED_RESOURCES,
)

GATHERABLE_RESOURCES: tuple[ResourceType, ...] = (
    ResourceType.LEAVES,
    ResourceType.WATER,
    ResourceType.PROTEIN,
    ResourceType.WASTE,
)

CARRIABLE_RESOURCES: tuple[ResourceType, ...] = (
    *GATHERABLE_RESOURCES,
    *PROCESSED_LOGISTICS_RESOURCES,
)

DEMAND_TRACKED_RESOURCES: tuple[ResourceType, ...] = (
    *GATHERABLE_RESOURCES,
    *PROCESSED_LOGISTICS_RESOURCES,
)

RESOURCE_PHEROMONES: dict[ResourceType, PheromoneType] = {
    ResourceType.LEAVES: PheromoneType.FOOD,
    ResourceType.SEEDS: PheromoneType.FOOD,
    ResourceType.WATER: PheromoneType.WATER,
    ResourceType.PROTEIN: PheromoneType.PROTEIN,
    ResourceType.WOOD_FIBER: PheromoneType.CONSTRUCTION,
    ResourceType.MINERALS: PheromoneType.CONSTRUCTION,
    ResourceType.SOIL: PheromoneType.CONSTRUCTION,
    ResourceType.RESIN: PheromoneType.CONSTRUCTION,
    ResourceType.DEAD_INSECTS: PheromoneType.PROTEIN,
    ResourceType.DEAD_ANTS: PheromoneType.PROTEIN,
    ResourceType.WASTE: PheromoneType.WASTE,
}

RESOURCE_TRAIL_WEIGHTS: dict[ResourceType, float] = {
    ResourceType.LEAVES: 1.0,
    ResourceType.SEEDS: 0.82,
    ResourceType.WATER: 1.12,
    ResourceType.PROTEIN: 1.08,
    ResourceType.WOOD_FIBER: 0.54,
    ResourceType.MINERALS: 0.46,
    ResourceType.SOIL: 0.42,
    ResourceType.RESIN: 0.58,
    ResourceType.DEAD_INSECTS: 0.86,
    ResourceType.DEAD_ANTS: 0.62,
    ResourceType.WASTE: 0.72,
}
