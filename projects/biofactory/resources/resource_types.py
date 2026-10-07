from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResourceType(str, Enum):
    LEAVES = "leaves"
    SEEDS = "seeds"
    PROTEIN = "protein"
    WATER = "water"
    WOOD_FIBER = "wood_fiber"
    MINERALS = "minerals"
    SOIL = "soil"
    RESIN = "resin"
    DEAD_INSECTS = "dead_insects"
    DEAD_ANTS = "dead_ants"
    WASTE = "waste"
    FOOD = "food"
    FUNGUS_SUBSTRATE = "fungus_substrate"
    FUNGUS = "fungus"
    NUTRIENT_PASTE = "nutrient_paste"
    PROTEIN_PASTE = "protein_paste"
    LARVAE_FOOD = "larvae_food"
    REINFORCED_SOIL = "reinforced_soil"
    STRUCTURAL_RESIN = "structural_resin"
    COMPOST = "compost"
    FERTILIZER = "fertilizer"
    SOLDIER_FEED = "soldier_feed"


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    kind: str
    weight: float
    decay_rate: float
    nutritional_value: float
    storage_requirement: str
    processing_requirement: str
    scent_radius: int
    color: tuple[int, int, int]


RESOURCE_SPECS: dict[ResourceType, ResourceSpec] = {
    ResourceType.LEAVES: ResourceSpec(
        "Leaves",
        "raw",
        1.0,
        0.0002,
        0.12,
        "dry storage",
        "fungus substrate or simplified food",
        7,
        (94, 210, 84),
    ),
    ResourceType.SEEDS: ResourceSpec(
        "Seeds",
        "raw",
        0.65,
        0.0003,
        0.45,
        "dry seed storage",
        "future food reserve / planting / soldier feed additive",
        5,
        (216, 184, 82),
    ),
    ResourceType.PROTEIN: ResourceSpec(
        "Protein",
        "raw",
        1.2,
        0.0010,
        0.85,
        "protein storage",
        "protein paste",
        6,
        (218, 92, 118),
    ),
    ResourceType.WATER: ResourceSpec(
        "Water",
        "raw",
        1.0,
        0.0000,
        0.2,
        "reservoir",
        "farms, nursery, queen chamber",
        5,
        (72, 158, 235),
    ),
    ResourceType.WOOD_FIBER: ResourceSpec(
        "Wood fiber",
        "raw",
        1.15,
        0.0001,
        0.02,
        "dry construction storage",
        "reinforced walls and structural resin",
        3,
        (150, 105, 63),
    ),
    ResourceType.MINERALS: ResourceSpec(
        "Minerals",
        "raw",
        1.45,
        0.0000,
        0.01,
        "mineral cache",
        "reinforced soil and chamber hardening",
        2,
        (148, 155, 150),
    ),
    ResourceType.SOIL: ResourceSpec(
        "Soil",
        "raw",
        1.30,
        0.0000,
        0.01,
        "loose soil pile",
        "construction, tunnel repair, compost bulk",
        2,
        (119, 88, 58),
    ),
    ResourceType.RESIN: ResourceSpec(
        "Resin",
        "raw",
        0.95,
        0.00015,
        0.02,
        "sealed resin cache",
        "structural resin and tunnel reinforcement",
        4,
        (225, 145, 59),
    ),
    ResourceType.DEAD_INSECTS: ResourceSpec(
        "Dead insects",
        "raw",
        1.25,
        0.0018,
        1.05,
        "cool protein storage",
        "protein paste / protein recovery",
        7,
        (172, 72, 92),
    ),
    ResourceType.DEAD_ANTS: ResourceSpec(
        "Dead ants",
        "raw",
        0.80,
        0.0015,
        0.65,
        "recovery chamber",
        "colony protein reuse",
        4,
        (126, 73, 82),
    ),
    ResourceType.WASTE: ResourceSpec(
        "Waste",
        "raw",
        0.8,
        0.0008,
        0.05,
        "waste chamber",
        "compost",
        4,
        (138, 96, 74),
    ),
    ResourceType.FOOD: ResourceSpec(
        "Food",
        "processed",
        0.7,
        0.0005,
        1.0,
        "food storage",
        "colony consumption",
        3,
        (220, 205, 120),
    ),
    ResourceType.FUNGUS_SUBSTRATE: ResourceSpec(
        "Fungus substrate",
        "processed",
        0.85,
        0.0006,
        0.18,
        "substrate bin",
        "fungus farm input",
        2,
        (130, 178, 88),
    ),
    ResourceType.FUNGUS: ResourceSpec(
        "Fungus",
        "processed",
        0.75,
        0.0007,
        0.9,
        "fungus storage",
        "nutrient processor input",
        2,
        (118, 226, 124),
    ),
    ResourceType.NUTRIENT_PASTE: ResourceSpec(
        "Nutrient paste",
        "processed",
        0.75,
        0.0007,
        1.15,
        "nutrient paste storage",
        "larvae food and worker brood",
        2,
        (232, 214, 122),
    ),
    ResourceType.PROTEIN_PASTE: ResourceSpec(
        "Protein paste",
        "processed",
        0.9,
        0.0006,
        1.25,
        "protein paste storage",
        "soldier feed / nursery precursor",
        2,
        (238, 112, 142),
    ),
    ResourceType.LARVAE_FOOD: ResourceSpec(
        "Larvae food",
        "processed",
        0.70,
        0.0008,
        1.35,
        "nursery food cache",
        "larvae growth",
        2,
        (247, 218, 145),
    ),
    ResourceType.REINFORCED_SOIL: ResourceSpec(
        "Reinforced soil",
        "processed",
        1.60,
        0.0000,
        0.0,
        "construction yard",
        "reinforced tunnel walls",
        1,
        (116, 103, 83),
    ),
    ResourceType.STRUCTURAL_RESIN: ResourceSpec(
        "Structural resin",
        "processed",
        1.10,
        0.00005,
        0.0,
        "sealed construction cache",
        "defensive infrastructure and wall bonding",
        1,
        (232, 168, 76),
    ),
    ResourceType.COMPOST: ResourceSpec(
        "Compost",
        "processed",
        0.95,
        0.0004,
        0.25,
        "compost storage",
        "fungus farm boost",
        2,
        (126, 100, 58),
    ),
    ResourceType.FERTILIZER: ResourceSpec(
        "Fertilizer",
        "processed",
        0.90,
        0.0004,
        0.30,
        "farm amendment cache",
        "fungus growth boost",
        2,
        (104, 132, 58),
    ),
    ResourceType.SOLDIER_FEED: ResourceSpec(
        "Soldier feed",
        "processed",
        0.95,
        0.0007,
        1.45,
        "barracks feed cache",
        "soldier larvae",
        2,
        (214, 83, 92),
    ),
}
