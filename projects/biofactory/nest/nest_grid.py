from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class NestCellType(IntEnum):
    WALL = 0
    TUNNEL = 1
    ENTRANCE = 2
    GENERAL_STORAGE = 3
    FOOD_STORAGE = 4
    WATER_RESERVOIR = 5
    PROTEIN_STORAGE = 6
    FUNGUS_FARM = 7
    NURSERY = 8
    BARRACKS = 9
    QUEEN_CHAMBER = 10
    WASTE_CHAMBER = 11
    COMPOST_CHAMBER = 12
    PROCESSING_CHAMBER = 13
    DEFENSIVE_CHOKE_POINT = 14
    TRAFFIC_HUB = 15
    EMERGENCY_TUNNEL = 16
    REINFORCED_WALL = 17


@dataclass
class NestGrid:
    cells: np.ndarray

    @classmethod
    def create(cls, width: int, height: int) -> "NestGrid":
        return cls(cells=np.full((height, width), NestCellType.WALL, dtype=np.uint8))
