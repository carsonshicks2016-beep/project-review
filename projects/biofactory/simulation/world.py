from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.random import Generator

from biofactory.config import WorldConfig
from biofactory.nest.nest_grid import NestCellType
from biofactory.simulation.terrain import (
    TERRAIN_PROPERTIES,
    TerrainProperties,
    TerrainType,
    construction_difficulty_map,
    energy_cost_map,
    generate_terrain,
    movement_speed_map,
    obstacle_map,
    pheromone_decay_map,
    pheromone_diffusion_map,
    resource_spawn_chance_map,
    visibility_map,
)


class SurfaceCellType(IntEnum):
    OPEN = 0
    NEST_ENTRANCE = 1
    SURFACE_STORAGE = 2


@dataclass
class World:
    width: int
    height: int
    terrain: np.ndarray
    obstacles: np.ndarray
    base_movement_speed: np.ndarray
    movement_speed: np.ndarray
    energy_cost: np.ndarray
    visibility: np.ndarray
    construction_difficulty: np.ndarray
    resource_spawn_chance: np.ndarray
    base_pheromone_decay_multiplier: np.ndarray
    pheromone_decay_multiplier: np.ndarray
    base_pheromone_diffusion_multiplier: np.ndarray
    pheromone_diffusion_multiplier: np.ndarray
    temperature: np.ndarray
    humidity: np.ndarray
    light: np.ndarray
    surface: np.ndarray
    underground: np.ndarray

    @classmethod
    def create(cls, config: WorldConfig, rng: Generator) -> "World":
        terrain = generate_terrain(config.width, config.height, rng, config.nest_x, config.nest_y)
        obstacles = obstacle_map(terrain)
        base_movement = movement_speed_map(terrain)
        base_decay = pheromone_decay_map(terrain)
        base_diffusion = pheromone_diffusion_map(terrain)
        surface = _surface_layer(config)
        underground = _underground_nest_layer(config)
        world = cls(
            width=config.width,
            height=config.height,
            terrain=terrain,
            obstacles=obstacles,
            base_movement_speed=base_movement,
            movement_speed=base_movement.copy(),
            energy_cost=energy_cost_map(terrain),
            visibility=visibility_map(terrain),
            construction_difficulty=construction_difficulty_map(terrain),
            resource_spawn_chance=resource_spawn_chance_map(terrain),
            base_pheromone_decay_multiplier=base_decay,
            pheromone_decay_multiplier=base_decay.copy(),
            base_pheromone_diffusion_multiplier=base_diffusion,
            pheromone_diffusion_multiplier=base_diffusion.copy(),
            temperature=np.full((config.height, config.width), 0.55, dtype=np.float32),
            humidity=np.full((config.height, config.width), 0.45, dtype=np.float32),
            light=np.full((config.height, config.width), 0.80, dtype=np.float32),
            surface=surface,
            underground=underground,
        )
        world.update_terrain_effects()
        return world

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def cell(self, x: float, y: float) -> tuple[int, int]:
        cx = min(self.width - 1, max(0, int(x)))
        cy = min(self.height - 1, max(0, int(y)))
        return cx, cy

    def is_blocked(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return True
        cx, cy = self.cell(x, y)
        return bool(self.obstacles[cy, cx])

    def terrain_at(self, x: float, y: float) -> TerrainType:
        cx, cy = self.cell(x, y)
        return TerrainType(int(self.terrain[cy, cx]))

    def terrain_properties_at(self, x: float, y: float) -> TerrainProperties:
        return TERRAIN_PROPERTIES[self.terrain_at(x, y)]

    def movement_speed_at(self, x: float, y: float) -> float:
        cx, cy = self.cell(x, y)
        return float(self.movement_speed[cy, cx])

    def energy_cost_at(self, x: float, y: float) -> float:
        cx, cy = self.cell(x, y)
        return float(self.energy_cost[cy, cx])

    def visibility_at(self, x: float, y: float) -> float:
        cx, cy = self.cell(x, y)
        return float(self.visibility[cy, cx])

    def construction_difficulty_at(self, x: float, y: float) -> float:
        cx, cy = self.cell(x, y)
        return float(self.construction_difficulty[cy, cx])

    def resource_spawn_chance_at(self, x: float, y: float) -> float:
        cx, cy = self.cell(x, y)
        return float(self.resource_spawn_chance[cy, cx])

    def environmental_inputs_at(self, x: float, y: float) -> tuple[float, float, float]:
        cx, cy = self.cell(x, y)
        return (
            float(self.temperature[cy, cx]),
            float(self.humidity[cy, cx]),
            float(self.light[cy, cx]),
        )

    def terrain_effects_at(self, x: float, y: float) -> dict[str, float]:
        cx, cy = self.cell(x, y)
        return {
            "movement_speed": float(self.movement_speed[cy, cx]),
            "energy_cost": float(self.energy_cost[cy, cx]),
            "pheromone_decay_multiplier": float(self.pheromone_decay_multiplier[cy, cx]),
            "pheromone_diffusion_multiplier": float(self.pheromone_diffusion_multiplier[cy, cx]),
            "visibility": float(self.visibility[cy, cx]),
            "construction_difficulty": float(self.construction_difficulty[cy, cx]),
            "resource_spawn_chance": float(self.resource_spawn_chance[cy, cx]),
        }

    def update_terrain_effects(self, weather: object | None = None, seasons: object | None = None, tick: int = 0) -> None:
        movement_multiplier = float(getattr(weather, "movement_multiplier", 1.0))
        decay_multiplier = float(getattr(weather, "pheromone_decay_multiplier", 1.0))
        diffusion_multiplier = float(getattr(weather, "pheromone_diffusion_multiplier", 1.0))
        temperature_delta = float(getattr(weather, "temperature_delta", 0.0))
        humidity_delta = float(getattr(weather, "humidity_delta", 0.0))
        light_multiplier = float(getattr(weather, "light_multiplier", 1.0))

        season_temperature_delta = float(getattr(seasons, "temperature_delta", 0.0))
        season_humidity_delta = float(getattr(seasons, "humidity_delta", 0.0))
        season_light_multiplier = float(getattr(seasons, "light_multiplier", 1.0))

        self.movement_speed[...] = np.clip(self.base_movement_speed * movement_multiplier, 0.05, 1.6)
        self.pheromone_decay_multiplier[...] = np.clip(
            self.base_pheromone_decay_multiplier * decay_multiplier,
            0.1,
            3.0,
        )
        self.pheromone_diffusion_multiplier[...] = np.clip(
            self.base_pheromone_diffusion_multiplier * diffusion_multiplier,
            0.1,
            3.0,
        )

        terrain_humidity = 0.10 * (self.terrain == TerrainType.WATER) + 0.05 * (self.terrain == TerrainType.MUD)
        terrain_shadow = 0.22 * (self.terrain == TerrainType.FOREST) + 0.16 * (self.terrain == TerrainType.ROOT_MASS)
        subtle_day_pulse = 0.02 * np.sin(tick / 360.0) if tick else 0.0
        self.temperature[...] = np.clip(0.55 + season_temperature_delta + temperature_delta + subtle_day_pulse, 0.0, 1.0)
        self.humidity[...] = np.clip(0.45 + terrain_humidity + season_humidity_delta + humidity_delta, 0.0, 1.0)
        self.light[...] = np.clip((0.80 - terrain_shadow) * season_light_multiplier * light_multiplier, 0.0, 1.0)


def _surface_layer(config: WorldConfig) -> np.ndarray:
    surface = np.full((config.height, config.width), SurfaceCellType.OPEN, dtype=np.uint8)
    yy, xx = np.ogrid[: config.height, : config.width]
    storage_mask = (xx - config.nest_x) ** 2 + (yy - config.nest_y) ** 2 <= config.storage_radius**2
    surface[storage_mask] = SurfaceCellType.SURFACE_STORAGE
    cx = min(config.width - 1, max(0, int(config.nest_x)))
    cy = min(config.height - 1, max(0, int(config.nest_y)))
    surface[cy, cx] = SurfaceCellType.NEST_ENTRANCE
    return surface


def _underground_nest_layer(config: WorldConfig) -> np.ndarray:
    underground = np.full((config.height, config.width), NestCellType.WALL, dtype=np.uint8)
    yy, xx = np.ogrid[: config.height, : config.width]
    chamber_radius = max(3.0, config.storage_radius * 0.85)
    tunnel_radius = max(5.0, config.storage_radius * 1.35)
    tunnel_mask = (xx - config.nest_x) ** 2 + (yy - config.nest_y) ** 2 <= tunnel_radius**2
    chamber_mask = (xx - config.nest_x) ** 2 + (yy - config.nest_y) ** 2 <= chamber_radius**2
    underground[tunnel_mask] = NestCellType.TUNNEL
    underground[chamber_mask] = NestCellType.QUEEN_CHAMBER
    cx = min(config.width - 1, max(0, int(config.nest_x)))
    cy = min(config.height - 1, max(0, int(config.nest_y)))
    underground[cy, cx] = NestCellType.ENTRANCE
    return underground
