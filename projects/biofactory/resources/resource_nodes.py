from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from biofactory.resources.resource_types import ResourceType


class ResourceSourceKind(str, Enum):
    LEAF_PATCH = "leaf_patch"
    SEED_PILE = "seed_pile"
    DEAD_INSECT = "dead_insect"
    WATER_POOL = "water_pool"
    MINERAL_DEPOSIT = "mineral_deposit"
    FALLEN_FRUIT = "fallen_fruit"
    TREE_ROOT = "tree_root"
    RIVAL_WASTE_DUMP = "rival_waste_dump"
    PREDATOR_CORPSE = "predator_corpse"
    SEASONAL_BURST = "seasonal_burst"
    SOIL_BANK = "soil_bank"
    RESIN_PATCH = "resin_patch"


@dataclass
class ResourceNode:
    resource_type: ResourceType
    x: float
    y: float
    amount: float
    source_kind: ResourceSourceKind = ResourceSourceKind.LEAF_PATCH
    radius: float = 1.0
    risk: float = 0.0
    renewable: bool = True
