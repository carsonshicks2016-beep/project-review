from __future__ import annotations

from dataclasses import dataclass, field

import math
import numpy as np

from biofactory.agents.ant import Ant
from biofactory.config import DEFAULT_CONFIG, SimulationConfig
from biofactory.colonies.colony import Colony
from biofactory.colonies.genetics import Genes
from biofactory.logistics.routing_metrics import RoutingMetrics
from biofactory.logistics.traffic import TrafficMetrics
from biofactory.neural.archive import BrainArchive
from biofactory.neural.brain import Brain
from biofactory.pheromones.pheromone_config import PheromoneType
from biofactory.pheromones.pheromone_layers import PheromoneLayers
from biofactory.resources.logistics import DEMAND_TRACKED_RESOURCES, GATHERABLE_RESOURCES, RAW_RESOURCE_TYPES
from biofactory.resources.resource_maps import ResourceMaps
from biofactory.resources.resource_types import ResourceType
from biofactory.simulation.events import EventLog
from biofactory.simulation.seasons import SeasonSystem
from biofactory.simulation.weather import WeatherSystem
from biofactory.simulation.world import World
from biofactory.utils.random_utils import make_rng
from biofactory.utils.spatial_hash import SpatialHash


@dataclass
class SimulationSnapshot:
    tick: int
    population: int
    leaves_world: float
    leaves_stored: float
    food_stored: float
    water_stored: float
    protein_stored: float
    waste_stored: float
    world_resources: dict[ResourceType, float]
    stored_resources: dict[ResourceType, float]
    deliveries: int
    demand_ratio: float
    max_food_pheromone: float
    max_demand_pheromone: float
    max_traffic_pheromone: float
    traffic: TrafficMetrics
    routing: RoutingMetrics


@dataclass
class SimulationEngine:
    config: SimulationConfig = DEFAULT_CONFIG
    tick: int = 0
    rng: np.random.Generator = field(init=False)
    world: World = field(init=False)
    resources: ResourceMaps = field(init=False)
    pheromones: PheromoneLayers = field(init=False)
    colony: Colony = field(init=False)
    ants: list[Ant] = field(init=False)
    spatial_hash: SpatialHash = field(init=False)
    ant_density: np.ndarray = field(init=False)
    ant_density_integral: np.ndarray = field(init=False)
    weather: WeatherSystem = field(default_factory=WeatherSystem)
    seasons: SeasonSystem = field(default_factory=SeasonSystem)
    events: EventLog = field(default_factory=EventLog)
    brain_archive: BrainArchive = field(default_factory=BrainArchive)
    routing_metrics: RoutingMetrics = field(default_factory=RoutingMetrics)
    traffic_metrics: TrafficMetrics = field(default_factory=TrafficMetrics)
    last_traffic_event_tick: int = -9999
    last_loop_stages: tuple[str, ...] = field(default_factory=tuple)
    statistics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = make_rng(self.config.world.seed)
        self.world = World.create(self.config.world, self.rng)
        self.resources = ResourceMaps.create(self.world, self.config.world, self.rng)
        self.pheromones = PheromoneLayers.create(self.world.width, self.world.height)
        self.colony = Colony.create_mvp(self.world.width, self.world.height, self.config.colony, self.config.world)
        self.spatial_hash = SpatialHash(cell_size=5.0)
        self.ant_density = np.zeros((self.world.height, self.world.width), dtype=np.uint16)
        self.ant_density_integral = np.zeros((self.world.height + 1, self.world.width + 1), dtype=np.uint32)
        self.ants = self._spawn_ants()
        self.colony.stats.ants_born = len(self.ants)
        self.events.add(0, "Colony founded")
        self.events.add(0, "MVP resource demand online")

    def step(self, ticks: int = 1) -> None:
        for _ in range(max(0, ticks)):
            stages: list[str] = []
            self.tick += 1

            stages.append("weather")
            self.weather.update(self.tick)

            stages.append("seasons")
            self.seasons.update(self.tick)

            stages.append("terrain_effects")
            self.world.update_terrain_effects(self.weather, self.seasons, self.tick)

            stages.append("pheromones")
            self.pheromones.update(self.world)

            stages.append("resource_generation")
            self.resources.regrow(self.world, self.rng, self.tick)

            for resource_type in GATHERABLE_RESOURCES:
                self.resources.prepare_scent_cache(resource_type)

            stages.append("production_chambers")
            births = self.colony.update(self.ants, self.pheromones, self.tick, self.events)

            stages.append("colonies")
            self._spawn_new_workers(births)

            stages.append("predators")
            self._update_predators()

            stages.append("ants")
            self.spatial_hash.clear()
            self.ant_density.fill(0)
            xs = np.fromiter((min(self.world.width - 1, max(0, int(ant.x))) for ant in self.ants), dtype=np.int32, count=len(self.ants))
            ys = np.fromiter((min(self.world.height - 1, max(0, int(ant.y))) for ant in self.ants), dtype=np.int32, count=len(self.ants))
            np.add.at(self.ant_density, (ys, xs), 1)
            self.ant_density_integral.fill(0)
            self.ant_density_integral[1:, 1:] = np.cumsum(np.cumsum(self.ant_density, axis=0), axis=1)
            for ant in self.ants:
                self.spatial_hash.insert(ant.ant_id, ant.x, ant.y)
            for ant in self.ants:
                ant.update(self)
            for resource_type in GATHERABLE_RESOURCES:
                self.resources.release_scent_cache(resource_type)

            stages.append("combat")
            self._update_combat()

            stages.append("logistics_metrics")
            self._update_metrics()

            stages.append("statistics")
            self._collect_statistics()

            stages.append("events")
            self._record_major_events()
            self.last_loop_stages = tuple(stages)

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            tick=self.tick,
            population=len(self.ants),
            leaves_world=self.resources.total(ResourceType.LEAVES),
            leaves_stored=self.colony.inventory.get(ResourceType.LEAVES),
            food_stored=self.colony.inventory.get(ResourceType.FOOD),
            water_stored=self.colony.inventory.get(ResourceType.WATER),
            protein_stored=self.colony.inventory.get(ResourceType.PROTEIN),
            waste_stored=self.colony.inventory.get(ResourceType.WASTE),
            world_resources={resource_type: self.resources.total(resource_type) for resource_type in RAW_RESOURCE_TYPES},
            stored_resources={
                resource_type: self.colony.inventory.get(resource_type)
                for resource_type in DEMAND_TRACKED_RESOURCES
            },
            deliveries=self.colony.stats.deliveries,
            demand_ratio=self.colony.demand_ratio(),
            max_food_pheromone=self.pheromones.max_value(PheromoneType.FOOD),
            max_demand_pheromone=self.pheromones.max_value(PheromoneType.DEMAND),
            max_traffic_pheromone=self.pheromones.max_value(PheromoneType.TRAFFIC),
            traffic=self.traffic_metrics,
            routing=self.routing_metrics,
        )

    def nearest_ant(self, x: float, y: float, max_distance: float = 3.0) -> Ant | None:
        best: Ant | None = None
        best_dist = max_distance
        for ant in self.ants:
            dist = math.hypot(ant.x - x, ant.y - y)
            if dist < best_dist:
                best = ant
                best_dist = dist
        return best

    def _spawn_ants(self) -> list[Ant]:
        base_brain = Brain.seeded_worker(self.rng)
        ants: list[Ant] = []
        for ant_id in range(self.config.ants.count):
            genes = Genes.random(self.rng)
            brain = base_brain.copy_mutated(self.rng, genes.mutation_rate)
            radius = float(self.rng.uniform(0.0, self.config.ants.spawn_radius))
            angle = float(self.rng.uniform(-math.pi, math.pi))
            x = self.config.world.nest_x + math.cos(angle) * radius
            y = self.config.world.nest_y + math.sin(angle) * radius
            ants.append(Ant.spawn(ant_id, self.colony.colony_id, x, y, self.rng, brain, genes))
        return ants

    def _spawn_new_workers(self, count: int) -> None:
        if count <= 0:
            return
        base_brain = self.brain_archive.best_for_role("Worker")
        if base_brain is None:
            base_brain = self.ants[0].brain if self.ants else Brain.seeded_worker(self.rng)
        next_id = max((ant.ant_id for ant in self.ants), default=-1) + 1
        sx, sy = self._brood_spawn_point()
        for offset in range(count):
            if len(self.ants) >= self.config.colony.max_population:
                break
            genes = Genes.random(self.rng)
            brain = base_brain.copy_mutated(self.rng, genes.mutation_rate)
            radius = float(self.rng.uniform(0.0, max(1.0, self.config.world.storage_radius * 0.35)))
            angle = float(self.rng.uniform(-math.pi, math.pi))
            x = max(0.0, min(self.world.width - 0.001, sx + math.cos(angle) * radius))
            y = max(0.0, min(self.world.height - 0.001, sy + math.sin(angle) * radius))
            self.ants.append(Ant.spawn(next_id + offset, self.colony.colony_id, x, y, self.rng, brain, genes))
            self.colony.stats.ants_born += 1

    def _brood_spawn_point(self) -> tuple[float, float]:
        nurseries = self.colony.nursery_chambers()
        if not nurseries:
            return self.colony.nest_x, self.colony.nest_y
        return nurseries[0].x, nurseries[0].y

    def _update_metrics(self) -> None:
        traffic_layer = self.pheromones.layers[PheromoneType.TRAFFIC]
        self.traffic_metrics = TrafficMetrics.from_layer(traffic_layer)
        delivery_ticks = [ant.last_delivery_ticks for ant in self.ants if ant.last_delivery_ticks > 0]
        avg_delivery = float(np.mean(delivery_ticks)) if delivery_ticks else 0.0
        self.routing_metrics = RoutingMetrics(
            average_delivery_ticks=avg_delivery,
            delivery_rate_per_10s=self.colony.stats.delivery_rate(self.tick),
            lost_cargo=0.0,
        )

    def _update_predators(self) -> None:
        # Phase 1 owns the loop hook; predator behavior arrives in a later move.
        return

    def _update_combat(self) -> None:
        # Phase 1 owns the loop hook; colony warfare arrives in a later move.
        return

    def _collect_statistics(self) -> None:
        self.statistics = {
            "tick": float(self.tick),
            "population": float(len(self.ants)),
            "deliveries": float(self.colony.stats.deliveries),
            "demand_ratio": float(self.colony.demand_ratio()),
            "traffic_cells": float(self.traffic_metrics.congested_cells),
            "max_traffic": float(self.traffic_metrics.max_traffic),
        }

    def _record_major_events(self) -> None:
        if self.tick % 900 == 0 and self.colony.stats.deliveries > 0:
            self.events.add(self.tick, f"Resource flow stable: {self.colony.stats.deliveries} deliveries")
        traffic_event_ready = self.tick - self.last_traffic_event_tick >= 1200
        severe_congestion = self.traffic_metrics.congested_cells > 120 or self.traffic_metrics.max_traffic > 58.0
        if self.tick % 300 == 0 and traffic_event_ready and severe_congestion:
            self.colony.stats.traffic_jam_events += 1
            self.last_traffic_event_tick = self.tick
            self.events.add(self.tick, "Traffic jam detected", severity="warning")
