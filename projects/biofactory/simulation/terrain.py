from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.random import Generator


class TerrainType(IntEnum):
    DIRT = 0
    GRASS = 1
    FOREST = 2
    SAND = 3
    WATER = 4
    ROCK = 5
    MUD = 6
    ROOT_MASS = 7
    GRAVEL = 8
    FUNGAL_SOIL = 9


@dataclass(frozen=True)
class TerrainProperties:
    name: str
    movement_speed: float
    energy_cost: float
    pheromone_decay_multiplier: float
    pheromone_diffusion_multiplier: float
    visibility: float
    construction_difficulty: float
    resource_spawn_chance: float
    blocks_movement: bool = False


TERRAIN_PROPERTIES: dict[TerrainType, TerrainProperties] = {
    TerrainType.DIRT: TerrainProperties("Dirt", 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.70),
    TerrainType.GRASS: TerrainProperties("Grass", 0.94, 1.03, 1.05, 1.05, 0.96, 1.10, 1.15),
    TerrainType.FOREST: TerrainProperties("Forest", 0.72, 1.22, 0.92, 0.75, 0.60, 1.45, 1.65),
    TerrainType.SAND: TerrainProperties("Sand", 0.70, 1.35, 1.22, 1.18, 1.05, 1.25, 0.42),
    TerrainType.WATER: TerrainProperties("Water", 0.15, 2.40, 1.70, 1.45, 0.88, 9.00, 0.10, True),
    TerrainType.ROCK: TerrainProperties("Rock", 0.25, 1.85, 0.82, 0.45, 0.70, 6.00, 0.18, True),
    TerrainType.MUD: TerrainProperties("Mud", 0.55, 1.55, 1.35, 0.95, 0.85, 1.70, 0.75),
    TerrainType.ROOT_MASS: TerrainProperties("Root mass", 0.62, 1.28, 0.78, 0.62, 0.48, 2.25, 1.45),
    TerrainType.GRAVEL: TerrainProperties("Gravel", 0.84, 1.16, 0.96, 0.82, 0.98, 1.85, 0.35),
    TerrainType.FUNGAL_SOIL: TerrainProperties("Fungal soil", 0.88, 0.92, 0.72, 0.65, 0.78, 0.80, 1.30),
}


def generate_terrain(width: int, height: int, rng: Generator, nest_x: float, nest_y: float) -> np.ndarray:
    terrain = np.full((height, width), TerrainType.GRASS, dtype=np.uint8)

    base_noise = rng.random((height, width))
    terrain[base_noise < 0.20] = TerrainType.DIRT
    terrain[(base_noise >= 0.20) & (base_noise < 0.30)] = TerrainType.SAND
    terrain[(base_noise >= 0.30) & (base_noise < 0.38)] = TerrainType.GRAVEL

    _paint_blobs(terrain, rng, TerrainType.FOREST, count=18, radius_range=(5, 17))
    _paint_blobs(terrain, rng, TerrainType.MUD, count=8, radius_range=(4, 11))
    _paint_blobs(terrain, rng, TerrainType.ROOT_MASS, count=9, radius_range=(3, 9))
    _paint_blobs(terrain, rng, TerrainType.FUNGAL_SOIL, count=5, radius_range=(3, 8))
    _paint_blobs(terrain, rng, TerrainType.WATER, count=5, radius_range=(4, 9))
    _paint_blobs(terrain, rng, TerrainType.ROCK, count=7, radius_range=(3, 8))

    _clear_spawn_area(terrain, nest_x, nest_y, radius=13)
    return terrain


def obstacle_map(terrain: np.ndarray) -> np.ndarray:
    obstacles = np.zeros_like(terrain, dtype=bool)
    for terrain_type, props in TERRAIN_PROPERTIES.items():
        if props.blocks_movement:
            obstacles[terrain == terrain_type] = True
    return obstacles


def movement_speed_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "movement_speed")


def energy_cost_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "energy_cost")


def pheromone_decay_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "pheromone_decay_multiplier")


def pheromone_diffusion_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "pheromone_diffusion_multiplier")


def visibility_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "visibility")


def construction_difficulty_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "construction_difficulty")


def resource_spawn_chance_map(terrain: np.ndarray) -> np.ndarray:
    return _property_map(terrain, "resource_spawn_chance")


def _property_map(terrain: np.ndarray, property_name: str) -> np.ndarray:
    values = np.ones_like(terrain, dtype=np.float32)
    for terrain_type, props in TERRAIN_PROPERTIES.items():
        values[terrain == terrain_type] = float(getattr(props, property_name))
    return values


def _paint_blobs(
    terrain: np.ndarray,
    rng: Generator,
    terrain_type: TerrainType,
    count: int,
    radius_range: tuple[int, int],
) -> None:
    height, width = terrain.shape
    yy, xx = np.ogrid[:height, :width]
    for _ in range(count):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(radius_range[0], radius_range[1] + 1))
        ry = max(2, int(rx * rng.uniform(0.55, 1.45)))
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        terrain[mask] = terrain_type


def _clear_spawn_area(terrain: np.ndarray, nest_x: float, nest_y: float, radius: int) -> None:
    height, width = terrain.shape
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - nest_x) ** 2 + (yy - nest_y) ** 2 <= radius * radius
    terrain[mask] = TerrainType.DIRT
