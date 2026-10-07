from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygame

from biofactory.rendering.colors import CARGO_LEAF, TERRAIN_COLORS
from biofactory.simulation.terrain import TerrainType


@dataclass
class ProceduralArtAssets:
    terrain: np.ndarray
    seed: int
    art_cell_size: int
    terrain_rgb: np.ndarray = field(init=False)
    leaf_noise: np.ndarray = field(init=False)
    _glow_cache: dict[tuple[tuple[int, int, int], int], pygame.Surface] = field(default_factory=dict)
    _glint_cache: dict[tuple[tuple[int, int, int], int], pygame.Surface] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed + 7919)
        self.terrain_rgb = self._build_terrain_texture(rng)
        height, width = self.terrain.shape
        self.leaf_noise = rng.uniform(
            0.74,
            1.26,
            (height * self.art_cell_size, width * self.art_cell_size),
        ).astype(np.float32)

    def upsample_scalar(self, values: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(values, self.art_cell_size, axis=0), self.art_cell_size, axis=1)

    def leaf_alpha(self, leaves: np.ndarray, scale: float) -> np.ndarray:
        strength = np.sqrt(np.clip(leaves / 15.0, 0.0, 1.0))
        alpha = self.upsample_scalar(strength) * self.leaf_noise * scale
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def leaf_color_array(self, leaves: np.ndarray) -> np.ndarray:
        high = self.upsample_scalar(np.clip(leaves / 15.0, 0.0, 1.0))
        color = np.array(CARGO_LEAF, dtype=np.float32)
        warm = np.array((205, 222, 94), dtype=np.float32)
        mix = np.clip(high * self.leaf_noise, 0.0, 1.0)[..., None]
        return np.clip(color * (1.0 - mix * 0.35) + warm * (mix * 0.35), 0, 255).astype(np.uint8)

    def glow_sprite(self, color: tuple[int, int, int], radius: int) -> pygame.Surface:
        radius = max(2, int(radius))
        key = (color, radius)
        cached = self._glow_cache.get(key)
        if cached is not None:
            return cached

        size = radius * 2 + 1
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2) / max(1, radius)
        falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2.2
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[:] = color
        alpha = np.clip(falloff * 255.0, 0, 255).astype(np.uint8)
        surface = surface_from_rgba(rgb, alpha)
        self._glow_cache[key] = surface
        return surface

    def cargo_glint(self, radius: int, color: tuple[int, int, int] = CARGO_LEAF) -> pygame.Surface:
        radius = max(2, int(radius))
        key = (color, radius)
        cached = self._glint_cache.get(key)
        if cached is not None:
            return cached
        size = radius * 2 + 1
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2) / max(1, radius)
        falloff = np.clip(1.0 - dist, 0.0, 1.0)
        alpha = np.clip(falloff**1.7 * 240.0, 0, 255).astype(np.uint8)
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[:] = color
        surface = surface_from_rgba(rgb, alpha)
        self._glint_cache[key] = surface
        return surface

    def _build_terrain_texture(self, rng: np.random.Generator) -> np.ndarray:
        base_cell = np.zeros((*self.terrain.shape, 3), dtype=np.float32)
        for terrain_type, color in TERRAIN_COLORS.items():
            base_cell[self.terrain == terrain_type] = color

        rgb = np.repeat(np.repeat(base_cell, self.art_cell_size, axis=0), self.art_cell_size, axis=1)
        high_height, high_width = rgb.shape[:2]
        cell_variation = rng.uniform(0.86, 1.14, self.terrain.shape).astype(np.float32)
        cell_variation = self.upsample_scalar(cell_variation)
        fine_noise = rng.normal(0.0, 1.0, (high_height, high_width)).astype(np.float32)
        brightness = np.clip(cell_variation + fine_noise * 0.035, 0.72, 1.28)[..., None]
        rgb *= brightness

        high_terrain = self.upsample_scalar(self.terrain)
        yy, xx = np.mgrid[:high_height, :high_width]

        forest = high_terrain == TerrainType.FOREST
        forest_mottle = (np.sin(xx * 0.17) + np.cos(yy * 0.13) + fine_noise) > 1.0
        rgb[forest & forest_mottle] += np.array((7, 28, 11), dtype=np.float32)

        grass = high_terrain == TerrainType.GRASS
        grass_specks = rng.random((high_height, high_width)) > 0.965
        rgb[grass & grass_specks] += np.array((38, 52, 16), dtype=np.float32)

        root = high_terrain == TerrainType.ROOT_MASS
        root_fibers = np.sin((xx + yy) * 0.23) > 0.68
        rgb[root & root_fibers] += np.array((32, 18, 8), dtype=np.float32)

        water = high_terrain == TerrainType.WATER
        shimmer = (np.sin(xx * 0.28 + yy * 0.05) > 0.78) | (rng.random((high_height, high_width)) > 0.985)
        rgb[water & shimmer] += np.array((20, 40, 64), dtype=np.float32)

        grains = rng.random((high_height, high_width)) > 0.955
        granular = (
            (high_terrain == TerrainType.DIRT)
            | (high_terrain == TerrainType.SAND)
            | (high_terrain == TerrainType.GRAVEL)
            | (high_terrain == TerrainType.MUD)
            | (high_terrain == TerrainType.ROCK)
        )
        rgb[granular & grains] += np.array((31, 27, 19), dtype=np.float32)

        fungal = high_terrain == TerrainType.FUNGAL_SOIL
        fungal_specks = rng.random((high_height, high_width)) > 0.970
        rgb[fungal & fungal_specks] += np.array((35, 24, 48), dtype=np.float32)

        return np.clip(rgb, 0, 255).astype(np.uint8)


def surface_from_rgb(rgb: np.ndarray) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return surface.convert()


def surface_from_rgba(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    height, width = alpha.shape
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pixels = pygame.surfarray.pixels3d(surface)
    pixels[:] = np.transpose(rgb, (1, 0, 2))
    del pixels
    pixels_alpha = pygame.surfarray.pixels_alpha(surface)
    pixels_alpha[:] = np.transpose(alpha, (1, 0))
    del pixels_alpha
    return surface
