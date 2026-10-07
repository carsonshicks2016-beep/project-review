from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class PheromoneType(IntEnum):
    FOOD = 0
    DEMAND = 1
    TRAFFIC = 2
    WATER = 3
    PROTEIN = 4
    CONSTRUCTION = 5
    DANGER = 6
    RECRUITMENT = 7
    TERRITORY = 8
    WASTE = 9
    QUEEN = 10
    EMERGENCY = 11
    STORAGE_FULL = 12
    NEST_ENTRANCE = 13
    DEATH = 14


@dataclass(frozen=True)
class PheromoneSpec:
    name: str
    diffusion_rate: float
    decay_rate: float
    max_concentration: float
    color: tuple[int, int, int]
    weather_sensitivity: float = 1.0
    terrain_sensitivity: float = 1.0


PHEROMONE_SPECS: dict[PheromoneType, PheromoneSpec] = {
    PheromoneType.FOOD: PheromoneSpec("Food trail", 0.085, 0.018, 90.0, (58, 235, 102)),
    PheromoneType.DEMAND: PheromoneSpec("Demand", 0.060, 0.014, 110.0, (215, 156, 255)),
    PheromoneType.TRAFFIC: PheromoneSpec("Traffic", 0.040, 0.060, 65.0, (255, 146, 42)),
    PheromoneType.WATER: PheromoneSpec("Water trail", 0.070, 0.018, 80.0, (68, 160, 255)),
    PheromoneType.PROTEIN: PheromoneSpec("Protein trail", 0.070, 0.018, 80.0, (255, 88, 126)),
    PheromoneType.CONSTRUCTION: PheromoneSpec("Construction", 0.045, 0.020, 65.0, (244, 174, 55)),
    PheromoneType.DANGER: PheromoneSpec("Danger", 0.050, 0.030, 100.0, (255, 43, 51)),
    PheromoneType.RECRUITMENT: PheromoneSpec("Recruitment", 0.060, 0.025, 90.0, (255, 225, 96)),
    PheromoneType.TERRITORY: PheromoneSpec("Territory", 0.025, 0.006, 70.0, (118, 192, 255)),
    PheromoneType.WASTE: PheromoneSpec("Waste", 0.040, 0.020, 70.0, (138, 96, 74)),
    PheromoneType.QUEEN: PheromoneSpec("Queen signal", 0.030, 0.010, 100.0, (255, 220, 235)),
    PheromoneType.EMERGENCY: PheromoneSpec("Emergency", 0.120, 0.040, 100.0, (255, 255, 255)),
    PheromoneType.STORAGE_FULL: PheromoneSpec("Storage full", 0.045, 0.020, 70.0, (160, 230, 255)),
    PheromoneType.NEST_ENTRANCE: PheromoneSpec("Nest entrance", 0.020, 0.006, 90.0, (210, 210, 160)),
    PheromoneType.DEATH: PheromoneSpec("Death", 0.050, 0.025, 80.0, (180, 32, 48)),
}


MVP_PHEROMONES = (
    PheromoneType.FOOD,
    PheromoneType.WATER,
    PheromoneType.PROTEIN,
    PheromoneType.CONSTRUCTION,
    PheromoneType.WASTE,
    PheromoneType.DEMAND,
    PheromoneType.TRAFFIC,
)
