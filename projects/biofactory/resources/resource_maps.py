from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from biofactory.config import WorldConfig
from biofactory.resources.logistics import GATHERABLE_RESOURCES, RAW_RESOURCE_TYPES
from biofactory.resources.resource_nodes import ResourceNode, ResourceSourceKind
from biofactory.resources.resource_types import RESOURCE_SPECS, ResourceType
from biofactory.simulation.terrain import TerrainType
from biofactory.simulation.world import World


@dataclass
class ResourceMaps:
    width: int
    height: int
    layers: dict[ResourceType, np.ndarray] = field(default_factory=dict)
    risk_layers: dict[ResourceType, np.ndarray] = field(default_factory=dict)
    source_nodes: list[ResourceNode] = field(default_factory=list)
    scent_integrals: dict[ResourceType, np.ndarray] = field(default_factory=dict)
    locked_scent_caches: set[ResourceType] = field(default_factory=set)
    dirty_scent_caches: set[ResourceType] = field(default_factory=set)

    @classmethod
    def create(cls, world: World, config: WorldConfig, rng: Generator) -> "ResourceMaps":
        maps = cls(width=world.width, height=world.height)
        for resource_type in RAW_RESOURCE_TYPES:
            maps.layers[resource_type] = np.zeros((world.height, world.width), dtype=np.float32)
            maps.risk_layers[resource_type] = np.zeros((world.height, world.width), dtype=np.float32)
        maps._seed_leaf_patches(world, config, rng)
        maps._seed_seed_sources(world, config, rng)
        maps._seed_water_sources(world, config, rng)
        maps._seed_protein_sources(world, config, rng)
        maps._seed_dead_insect_sources(world, config, rng)
        maps._seed_construction_sources(world, config, rng)
        maps._seed_waste_sources(world, config, rng)
        maps._seed_rival_waste_dumps(world, config, rng)
        maps._seed_predator_corpses(world, config, rng)
        return maps

    def amount_at(self, resource_type: ResourceType, x: float, y: float) -> float:
        layer = self.layers[resource_type]
        cx, cy = self._cell(x, y)
        return float(layer[cy, cx])

    def total(self, resource_type: ResourceType) -> float:
        return float(np.sum(self.layers[resource_type]))

    def risk_at(self, resource_type: ResourceType, x: float, y: float) -> float:
        layer = self.risk_layers.get(resource_type)
        if layer is None:
            return 0.0
        cx, cy = self._cell(x, y)
        return float(layer[cy, cx])

    def source_nodes_for(self, resource_type: ResourceType | None = None) -> list[ResourceNode]:
        if resource_type is None:
            return list(self.source_nodes)
        return [node for node in self.source_nodes if node.resource_type == resource_type]

    def take_at(self, resource_type: ResourceType, x: float, y: float, amount: float) -> float:
        layer = self.layers[resource_type]
        cx, cy = self._cell(x, y)
        taken = min(float(layer[cy, cx]), amount)
        layer[cy, cx] -= taken
        self._mark_scent_dirty(resource_type)
        return taken

    def add_at(self, resource_type: ResourceType, x: float, y: float, amount: float) -> None:
        layer = self.layers.setdefault(
            resource_type,
            np.zeros((self.height, self.width), dtype=np.float32),
        )
        cx, cy = self._cell(x, y)
        layer[cy, cx] += amount
        self._mark_scent_dirty(resource_type)

    def scent_at(self, resource_type: ResourceType, x: float, y: float, radius: int) -> float:
        layer = self.layers[resource_type]
        cx, cy = self._cell(x, y)
        x0 = max(0, cx - radius)
        x1 = min(self.width, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(self.height, cy + radius + 1)
        integral = self.scent_integrals.get(resource_type)
        if integral is not None:
            total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            return float(total / max(1, radius * radius))
        window = layer[y0:y1, x0:x1]
        return float(np.sum(window) / max(1, radius * radius)) if window.size else 0.0

    def prepare_scent_cache(self, resource_type: ResourceType) -> None:
        layer = self.layers[resource_type]
        integral = np.zeros((self.height + 1, self.width + 1), dtype=np.float32)
        integral[1:, 1:] = np.cumsum(np.cumsum(layer, axis=0), axis=1)
        self.scent_integrals[resource_type] = integral
        self.locked_scent_caches.add(resource_type)
        self.dirty_scent_caches.discard(resource_type)

    def release_scent_cache(self, resource_type: ResourceType) -> None:
        self.locked_scent_caches.discard(resource_type)
        if resource_type in self.dirty_scent_caches:
            self.scent_integrals.pop(resource_type, None)
            self.dirty_scent_caches.discard(resource_type)

    def regrow(self, world: World, rng: Generator, tick: int) -> None:
        if tick % 60 == 0:
            self.apply_decay(60)
        if tick % 240 == 0:
            self._regrow_leaves(world, rng)
        if tick % 360 == 0:
            self._regrow_water(world, rng)
        if tick % 420 == 0:
            self._regrow_scattered(world, rng, ResourceType.SEEDS, 5, 0.5, 2.4, 12.0)
        if tick % 480 == 0:
            self._regrow_scattered(world, rng, ResourceType.PROTEIN, 2, 2.0, 6.0, 12.0)
            self._regrow_scattered(world, rng, ResourceType.DEAD_INSECTS, 2, 1.0, 3.5, 10.0)
        if tick % 600 == 0:
            self._regrow_scattered(world, rng, ResourceType.SOIL, 8, 0.8, 3.0, 14.0)
        if tick % 720 == 0:
            self._regrow_scattered(world, rng, ResourceType.WASTE, 2, 1.0, 3.2, 10.0)
        if tick % 900 == 0:
            self._regrow_scattered(world, rng, ResourceType.WOOD_FIBER, 3, 0.8, 2.5, 10.0)
            self._regrow_scattered(world, rng, ResourceType.RESIN, 2, 0.5, 1.8, 8.0)
        if tick % 1200 == 0:
            self._regrow_scattered(world, rng, ResourceType.MINERALS, 2, 0.8, 2.8, 10.0)
        if tick % 1800 == 0:
            self._seasonal_resource_burst(world, rng)

    def apply_decay(self, ticks: int = 1) -> None:
        for resource_type, layer in self.layers.items():
            decay_rate = RESOURCE_SPECS[resource_type].decay_rate
            if decay_rate <= 0.0:
                continue
            before = float(np.sum(layer))
            if before <= 0.0:
                continue
            layer *= max(0.0, (1.0 - decay_rate) ** ticks)
            if float(np.sum(layer)) < before:
                self._mark_scent_dirty(resource_type)

    def _mark_scent_dirty(self, resource_type: ResourceType) -> None:
        if resource_type in self.locked_scent_caches:
            self.dirty_scent_caches.add(resource_type)
        else:
            self.scent_integrals.pop(resource_type, None)

    def _seed_leaf_patches(self, world: World, config: WorldConfig, rng: Generator) -> None:
        leaves = self.layers[ResourceType.LEAVES]
        yy, xx = np.ogrid[:world.height, :world.width]
        placed = 0
        attempts = 0
        while placed < config.leaf_patch_count and attempts < config.leaf_patch_count * 12:
            attempts += 1
            cx = int(rng.integers(5, world.width - 5))
            cy = int(rng.integers(5, world.height - 5))
            if world.obstacles[cy, cx]:
                continue
            if (cx - config.nest_x) ** 2 + (cy - config.nest_y) ** 2 < 18 * 18:
                continue
            spawn_chance = float(world.resource_spawn_chance[cy, cx])
            if spawn_chance < 0.3:
                continue
            radius = int(rng.integers(config.leaf_patch_min_radius, config.leaf_patch_max_radius + 1))
            amount = config.leaf_patch_amount * spawn_chance * rng.uniform(0.65, 1.35)
            self._add_patch(
                world,
                ResourceType.LEAVES,
                cx,
                cy,
                radius,
                amount,
                max_amount=18.0,
                source_kind=ResourceSourceKind.LEAF_PATCH,
                base_risk=0.10,
                renewable=True,
            )
            placed += 1
        np.clip(leaves, 0.0, 18.0, out=leaves)

    def _seed_seed_sources(self, world: World, config: WorldConfig, rng: Generator) -> None:
        candidates = np.argwhere(
            ((world.terrain == TerrainType.GRASS) | (world.terrain == TerrainType.FOREST) | (world.terrain == TerrainType.FUNGAL_SOIL))
            & ~world.obstacles
        )
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.SEEDS,
            candidates,
            count=max(8, config.leaf_patch_count // 2),
            min_radius=1,
            max_radius=3,
            amount=3.2,
            max_amount=12.0,
            source_kind=ResourceSourceKind.SEED_PILE,
            base_risk=0.14,
        )
        fruit_count = max(3, config.leaf_patch_count // 8)
        for _ in range(fruit_count):
            if len(candidates) == 0:
                return
            cy, cx = candidates[int(rng.integers(0, len(candidates)))]
            if (cx - config.nest_x) ** 2 + (cy - config.nest_y) ** 2 < 16 * 16:
                continue
            radius = int(rng.integers(2, 5))
            amount = float(rng.uniform(2.0, 5.5))
            self._add_patch(world, ResourceType.SEEDS, cx, cy, radius, amount, 12.0, ResourceSourceKind.FALLEN_FRUIT, 0.18)
            self._add_patch(world, ResourceType.LEAVES, cx, cy, radius + 1, amount * 0.65, 18.0, ResourceSourceKind.FALLEN_FRUIT, 0.18)

    def _seed_water_sources(self, world: World, config: WorldConfig, rng: Generator) -> None:
        near_water = self._near_terrain(world, TerrainType.WATER)
        mud = world.terrain == TerrainType.MUD
        candidates = np.argwhere((near_water | mud) & ~world.obstacles & (world.terrain != TerrainType.WATER))
        count = max(12, config.leaf_patch_count // 2)
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.WATER,
            candidates,
            count=count,
            min_radius=1,
            max_radius=3,
            amount=11.0,
            max_amount=22.0,
            source_kind=ResourceSourceKind.WATER_POOL,
            base_risk=0.28,
        )

    def _seed_protein_sources(self, world: World, config: WorldConfig, rng: Generator) -> None:
        source_terrain = (
            (world.terrain == TerrainType.GRASS)
            | (world.terrain == TerrainType.FOREST)
            | (world.terrain == TerrainType.ROOT_MASS)
            | (world.terrain == TerrainType.FUNGAL_SOIL)
            | (world.terrain == TerrainType.MUD)
        )
        candidates = np.argwhere(source_terrain & ~world.obstacles)
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.PROTEIN,
            candidates,
            count=max(10, config.leaf_patch_count // 2),
            min_radius=1,
            max_radius=2,
            amount=5.2,
            max_amount=13.0,
            source_kind=ResourceSourceKind.DEAD_INSECT,
            base_risk=0.24,
        )

    def _seed_dead_insect_sources(self, world: World, config: WorldConfig, rng: Generator) -> None:
        source_terrain = (
            (world.terrain == TerrainType.GRASS)
            | (world.terrain == TerrainType.FOREST)
            | (world.terrain == TerrainType.ROOT_MASS)
            | (world.terrain == TerrainType.MUD)
        )
        candidates = np.argwhere(source_terrain & ~world.obstacles)
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.DEAD_INSECTS,
            candidates,
            count=max(8, config.leaf_patch_count // 3),
            min_radius=1,
            max_radius=2,
            amount=4.6,
            max_amount=11.0,
            source_kind=ResourceSourceKind.DEAD_INSECT,
            base_risk=0.30,
        )

    def _seed_construction_sources(self, world: World, config: WorldConfig, rng: Generator) -> None:
        root_candidates = np.argwhere(((world.terrain == TerrainType.FOREST) | (world.terrain == TerrainType.ROOT_MASS)) & ~world.obstacles)
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.WOOD_FIBER,
            root_candidates,
            count=max(8, config.leaf_patch_count // 3),
            min_radius=1,
            max_radius=3,
            amount=4.4,
            max_amount=11.0,
            source_kind=ResourceSourceKind.TREE_ROOT,
            base_risk=0.18,
        )
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.RESIN,
            root_candidates,
            count=max(6, config.leaf_patch_count // 4),
            min_radius=1,
            max_radius=2,
            amount=2.7,
            max_amount=8.0,
            source_kind=ResourceSourceKind.RESIN_PATCH,
            base_risk=0.20,
        )
        mineral_candidates = np.argwhere(
            ((world.terrain == TerrainType.GRAVEL) | self._near_terrain(world, TerrainType.ROCK))
            & ~world.obstacles
        )
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.MINERALS,
            mineral_candidates,
            count=max(7, config.leaf_patch_count // 3),
            min_radius=1,
            max_radius=3,
            amount=4.8,
            max_amount=12.0,
            source_kind=ResourceSourceKind.MINERAL_DEPOSIT,
            base_risk=0.24,
        )
        soil_candidates = np.argwhere(
            ((world.terrain == TerrainType.DIRT) | (world.terrain == TerrainType.MUD) | (world.terrain == TerrainType.SAND))
            & ~world.obstacles
        )
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.SOIL,
            soil_candidates,
            count=max(10, config.leaf_patch_count // 2),
            min_radius=1,
            max_radius=3,
            amount=5.4,
            max_amount=14.0,
            source_kind=ResourceSourceKind.SOIL_BANK,
            base_risk=0.12,
        )

    def _seed_waste_sources(self, world: World, config: WorldConfig, rng: Generator) -> None:
        source_terrain = (
            (world.terrain == TerrainType.MUD)
            | (world.terrain == TerrainType.ROOT_MASS)
            | (world.terrain == TerrainType.FUNGAL_SOIL)
            | (world.terrain == TerrainType.FOREST)
        )
        candidates = np.argwhere(source_terrain & ~world.obstacles)
        self._seed_blobs(
            world,
            config,
            rng,
            ResourceType.WASTE,
            candidates,
            count=max(6, config.leaf_patch_count // 4),
            min_radius=1,
            max_radius=2,
            amount=4.0,
            max_amount=10.0,
            source_kind=ResourceSourceKind.RIVAL_WASTE_DUMP,
            base_risk=0.34,
        )

    def _seed_rival_waste_dumps(self, world: World, config: WorldConfig, rng: Generator) -> None:
        candidates = np.argwhere(~world.obstacles & (world.resource_spawn_chance > 0.25))
        if len(candidates) == 0:
            return
        for _ in range(3):
            cy, cx = candidates[int(rng.integers(0, len(candidates)))]
            if (cx - config.nest_x) ** 2 + (cy - config.nest_y) ** 2 < 35 * 35:
                continue
            radius = int(rng.integers(2, 5))
            self._add_patch(world, ResourceType.WASTE, cx, cy, radius, float(rng.uniform(5.0, 10.0)), 10.0, ResourceSourceKind.RIVAL_WASTE_DUMP, 0.62, renewable=False)
            self._add_patch(world, ResourceType.DEAD_ANTS, cx, cy, radius, float(rng.uniform(2.0, 5.0)), 8.0, ResourceSourceKind.RIVAL_WASTE_DUMP, 0.72, renewable=False)

    def _seed_predator_corpses(self, world: World, config: WorldConfig, rng: Generator) -> None:
        candidates = np.argwhere(~world.obstacles & (world.light > 0.55) & (world.resource_spawn_chance > 0.20))
        if len(candidates) == 0:
            return
        for _ in range(max(3, config.leaf_patch_count // 8)):
            cy, cx = candidates[int(rng.integers(0, len(candidates)))]
            if (cx - config.nest_x) ** 2 + (cy - config.nest_y) ** 2 < 24 * 24:
                continue
            radius = int(rng.integers(1, 3))
            amount = float(rng.uniform(4.0, 9.0))
            self._add_patch(world, ResourceType.DEAD_INSECTS, cx, cy, radius, amount, 11.0, ResourceSourceKind.PREDATOR_CORPSE, 0.68, renewable=False)
            self._add_patch(world, ResourceType.PROTEIN, cx, cy, radius, amount * 0.45, 13.0, ResourceSourceKind.PREDATOR_CORPSE, 0.68, renewable=False)

    def _seed_blobs(
        self,
        world: World,
        config: WorldConfig,
        rng: Generator,
        resource_type: ResourceType,
        candidates: np.ndarray,
        *,
        count: int,
        min_radius: int,
        max_radius: int,
        amount: float,
        max_amount: float,
        source_kind: ResourceSourceKind,
        base_risk: float,
    ) -> None:
        if len(candidates) == 0:
            return
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 14:
            attempts += 1
            cy, cx = candidates[int(rng.integers(0, len(candidates)))]
            if (cx - config.nest_x) ** 2 + (cy - config.nest_y) ** 2 < 14 * 14:
                continue
            radius = int(rng.integers(min_radius, max_radius + 1))
            self._add_patch(
                world,
                resource_type,
                cx,
                cy,
                radius,
                amount * float(world.resource_spawn_chance[cy, cx]) * float(rng.uniform(0.55, 1.35)),
                max_amount,
                source_kind,
                base_risk,
            )
            placed += 1

    def _regrow_leaves(self, world: World, rng: Generator) -> None:
        layer = self.layers[ResourceType.LEAVES]
        candidates = np.argwhere((world.terrain == TerrainType.FOREST) | (world.terrain == TerrainType.GRASS))
        if len(candidates) == 0:
            return
        for _ in range(10):
            y, x = candidates[int(rng.integers(0, len(candidates)))]
            if world.obstacles[y, x]:
                continue
            chance = float(world.resource_spawn_chance[y, x])
            layer[y, x] += float(rng.uniform(0.2, 0.8) * chance)
        np.clip(layer, 0.0, 18.0, out=layer)
        self._mark_scent_dirty(ResourceType.LEAVES)

    def _regrow_water(self, world: World, rng: Generator) -> None:
        layer = self.layers[ResourceType.WATER]
        candidates = np.argwhere((self._near_terrain(world, TerrainType.WATER) | (world.terrain == TerrainType.MUD)) & ~world.obstacles)
        if len(candidates) == 0:
            return
        for _ in range(12):
            y, x = candidates[int(rng.integers(0, len(candidates)))]
            layer[y, x] += float(rng.uniform(0.8, 2.2))
        np.clip(layer, 0.0, 22.0, out=layer)
        self._mark_scent_dirty(ResourceType.WATER)

    def _regrow_scattered(
        self,
        world: World,
        rng: Generator,
        resource_type: ResourceType,
        count: int,
        min_amount: float,
        max_amount: float,
        cap: float,
    ) -> None:
        layer = self.layers[resource_type]
        candidates = np.argwhere(
            ~world.obstacles
            & (world.terrain != TerrainType.WATER)
            & (world.terrain != TerrainType.ROCK)
            & (world.resource_spawn_chance > 0.15)
        )
        if len(candidates) == 0:
            return
        for _ in range(count):
            y, x = candidates[int(rng.integers(0, len(candidates)))]
            layer[y, x] += float(rng.uniform(min_amount, max_amount) * world.resource_spawn_chance[y, x])
        np.clip(layer, 0.0, cap, out=layer)
        self._mark_scent_dirty(resource_type)

    def _seasonal_resource_burst(self, world: World, rng: Generator) -> None:
        candidates = np.argwhere(((world.terrain == TerrainType.FOREST) | (world.terrain == TerrainType.GRASS)) & ~world.obstacles)
        if len(candidates) == 0:
            return
        for _ in range(6):
            y, x = candidates[int(rng.integers(0, len(candidates)))]
            radius = int(rng.integers(2, 5))
            self._add_patch(
                world,
                ResourceType.SEEDS,
                int(x),
                int(y),
                radius,
                float(rng.uniform(2.5, 7.0)),
                12.0,
                ResourceSourceKind.SEASONAL_BURST,
                0.20,
            )
            self._add_patch(
                world,
                ResourceType.LEAVES,
                int(x),
                int(y),
                radius + 1,
                float(rng.uniform(4.0, 10.0)),
                18.0,
                ResourceSourceKind.SEASONAL_BURST,
                0.18,
            )

    def _add_patch(
        self,
        world: World,
        resource_type: ResourceType,
        cx: int,
        cy: int,
        radius: int,
        amount: float,
        max_amount: float,
        source_kind: ResourceSourceKind,
        base_risk: float,
        renewable: bool = True,
    ) -> None:
        if amount <= 0.0:
            return
        layer = self.layers[resource_type]
        risk_layer = self.risk_layers[resource_type]
        yy, xx = np.ogrid[:world.height, :world.width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        mask &= ~world.obstacles
        if not np.any(mask):
            return
        layer[mask] += amount
        np.clip(layer, 0.0, max_amount, out=layer)
        risk = self._source_risk(world, cx, cy, base_risk)
        risk_layer[mask] = np.maximum(risk_layer[mask], risk)
        self.source_nodes.append(
            ResourceNode(
                resource_type=resource_type,
                x=float(cx),
                y=float(cy),
                amount=float(amount),
                source_kind=source_kind,
                radius=float(radius),
                risk=risk,
                renewable=renewable,
            )
        )

    def _source_risk(self, world: World, cx: int, cy: int, base_risk: float) -> float:
        terrain_risk = min(0.25, max(0.0, world.energy_cost[cy, cx] - 1.0) * 0.10)
        exposure_risk = max(0.0, world.light[cy, cx] - 0.70) * 0.25
        water_risk = 0.18 if self._near_terrain(world, TerrainType.WATER)[cy, cx] else 0.0
        rock_risk = 0.10 if self._near_terrain(world, TerrainType.ROCK)[cy, cx] else 0.0
        return float(np.clip(base_risk + terrain_risk + exposure_risk + water_risk + rock_risk, 0.0, 1.0))

    def _near_terrain(self, world: World, terrain_type: TerrainType) -> np.ndarray:
        target = world.terrain == terrain_type
        near = np.zeros_like(target, dtype=bool)
        near[1:, :] |= target[:-1, :]
        near[:-1, :] |= target[1:, :]
        near[:, 1:] |= target[:, :-1]
        near[:, :-1] |= target[:, 1:]
        return near

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = min(self.width - 1, max(0, int(x)))
        cy = min(self.height - 1, max(0, int(y)))
        return cx, cy
