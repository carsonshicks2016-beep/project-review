from __future__ import annotations

import numpy as np

from biofactory.simulation.terrain import TerrainType


TERRAIN_COLORS: dict[TerrainType, tuple[int, int, int]] = {
    TerrainType.DIRT: (58, 49, 38),
    TerrainType.GRASS: (27, 82, 48),
    TerrainType.FOREST: (16, 58, 35),
    TerrainType.SAND: (111, 98, 67),
    TerrainType.WATER: (20, 58, 96),
    TerrainType.ROCK: (61, 65, 67),
    TerrainType.MUD: (49, 43, 36),
    TerrainType.ROOT_MASS: (45, 34, 28),
    TerrainType.GRAVEL: (72, 75, 70),
    TerrainType.FUNGAL_SOIL: (55, 47, 66),
}


BACKGROUND = (12, 14, 15)
PANEL_BG = (14, 18, 20)
PANEL_LINE = (58, 69, 72)
TEXT = (224, 233, 226)
TEXT_DIM = (157, 170, 164)
WARNING = (250, 188, 80)
STORAGE = (236, 230, 145)
ANT_OUTLINE = (4, 8, 10)
CARGO_LEAF = (124, 245, 96)
FOOD_STORE = (236, 196, 96)
DEMAND_CORE = (218, 158, 255)


def terrain_rgb_array(terrain: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*terrain.shape, 3), dtype=np.uint8)
    for terrain_type, color in TERRAIN_COLORS.items():
        rgb[terrain == terrain_type] = color
    return rgb


def blend_color(base: np.ndarray, color: tuple[int, int, int], alpha: np.ndarray) -> np.ndarray:
    alpha3 = np.clip(alpha, 0.0, 1.0)[..., None]
    color_array = np.array(color, dtype=np.float32)
    blended = base.astype(np.float32) * (1.0 - alpha3) + color_array * alpha3
    return np.clip(blended, 0, 255).astype(np.uint8)
