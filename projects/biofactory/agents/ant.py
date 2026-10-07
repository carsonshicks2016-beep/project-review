from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import math
import numpy as np

from biofactory.colonies.genetics import Genes
from biofactory.colonies.role_inference import infer_role
from biofactory.config import AntConfig
from biofactory.neural.brain import Brain, BrainInput, BrainOutput, INPUT_SIZE, MEMORY_SIZE, OUTPUT_SIZE
from biofactory.pheromones.pheromone_config import PheromoneType
from biofactory.resources.logistics import GATHERABLE_RESOURCES, PROCESSED_LOGISTICS_RESOURCES, RESOURCE_PHEROMONES, RESOURCE_TRAIL_WEIGHTS
from biofactory.resources.resource_types import RESOURCE_SPECS, ResourceType
from biofactory.utils.math_utils import angle_delta, angle_to, clamp, wrap_angle

if TYPE_CHECKING:
    from numpy.random import Generator
    from biofactory.simulation.engine import SimulationEngine


RESOURCE_TYPE_INPUT_VALUES = {
    resource_type: (index + 1) / max(1, len(ResourceType))
    for index, resource_type in enumerate(ResourceType)
}


@dataclass
class Ant:
    ant_id: int
    colony_id: int
    x: float
    y: float
    angle: float
    brain: Brain
    genes: Genes
    velocity: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    health: float = 1.0
    energy: float = 1.0
    age: float = 0.0
    lifespan: float = 25000.0
    inventory_type: ResourceType | None = None
    inventory_amount: float = 0.0
    memory: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_SIZE, dtype=np.float32))
    last_inputs: np.ndarray = field(default_factory=lambda: np.zeros(INPUT_SIZE, dtype=np.float32))
    last_outputs: np.ndarray = field(default_factory=lambda: np.zeros(OUTPUT_SIZE, dtype=np.float32))
    inferred_role: str = "Worker"
    deliveries: int = 0
    resources_delivered: float = 0.0
    resources_delivered_by_type: dict[ResourceType, float] = field(default_factory=dict)
    distance_traveled: float = 0.0
    distance_from_nest_max: float = 0.0
    carrying_ticks: int = 0
    target_resource: ResourceType = ResourceType.LEAVES
    target_chamber_id: int | None = None
    pickup_chamber_id: int | None = None
    cargo_source: str | None = None
    cargo_destination_kind: str | None = None
    kills: int = 0
    construction_contributed: float = 0.0
    digging_ticks: int = 0
    building_ticks: int = 0
    attack_intent_ticks: int = 0
    defend_ticks: int = 0
    grooming_ticks: int = 0
    allies_helped: float = 0.0
    fungus_tended: float = 0.0
    danger_pheromone_laid: float = 0.0
    recruitment_pheromone_laid: float = 0.0
    waste_removed: float = 0.0
    nursing_ticks: int = 0
    larvae_helped: float = 0.0
    last_pickup_tick: int = 0
    last_delivery_ticks: int = 0

    @property
    def has_cargo(self) -> bool:
        return self.inventory_type is not None and self.inventory_amount > 0.0

    @property
    def inventory_capacity(self) -> float:
        return max(0.35, self.genes.inventory_capacity)

    @property
    def aggression(self) -> float:
        return self.genes.aggression

    @property
    def curiosity(self) -> float:
        return self.genes.curiosity

    @property
    def fear(self) -> float:
        return self.genes.fear

    @property
    def strength(self) -> float:
        return self.genes.strength

    @property
    def speed(self) -> float:
        return self.genes.speed

    @property
    def vision_range(self) -> float:
        return self.genes.vision_range

    @property
    def pheromone_sensitivity(self) -> float:
        return self.genes.pheromone_sensitivity

    @property
    def traffic_tolerance(self) -> float:
        return self.genes.traffic_tolerance

    @property
    def lifetime_stats(self) -> dict[str, float]:
        return {
            "deliveries": float(self.deliveries),
            "resources_delivered": float(self.resources_delivered),
            "distance_traveled": float(self.distance_traveled),
            "kills": float(self.kills),
            "construction_contributed": float(self.construction_contributed),
            "larvae_helped": float(self.larvae_helped),
            "waste_removed": float(self.waste_removed),
            "fungus_tended": float(self.fungus_tended),
            "allies_helped": float(self.allies_helped),
        }

    @classmethod
    def spawn(cls, ant_id: int, colony_id: int, x: float, y: float, rng: "Generator", brain: Brain, genes: Genes) -> "Ant":
        return cls(
            ant_id=ant_id,
            colony_id=colony_id,
            x=x,
            y=y,
            angle=float(rng.uniform(-math.pi, math.pi)),
            brain=brain,
            genes=genes,
            health=float(min(1.0, max(0.25, genes.health))),
            lifespan=float(25000.0 * genes.lifespan),
        )

    def update(self, engine: "SimulationEngine") -> None:
        cfg = engine.config.ants
        colony = engine.colony
        world = engine.world
        pheromones = engine.pheromones
        resources = engine.resources
        rng = engine.rng
        cx, cy = world.cell(self.x, self.y)

        self.age += 1.0
        if self.has_cargo:
            self.carrying_ticks += 1

        if self.has_cargo:
            target_resource = self.inventory_type
        else:
            needs_target = (
                self.age <= 1.0
                or self.target_resource not in GATHERABLE_RESOURCES
                or int(self.age + self.ant_id) % (30 + self.ant_id % 21) == 0
                or colony.resource_priority(self.target_resource) < 0.04
            )
            if needs_target:
                self.target_resource = self._target_resource(engine, cfg)
            target_resource = self.target_resource
        demand_here = float(pheromones.layers[PheromoneType.DEMAND][cy, cx])
        traffic_here = float(pheromones.layers[PheromoneType.TRAFFIC][cy, cx])
        visibility = float(world.visibility[cy, cx])
        if target_resource in GATHERABLE_RESOURCES:
            layer = resources.layers.get(target_resource)
            resource_here = float(layer[cy, cx]) if layer is not None else 0.0
        else:
            resource_here = 0.0
        temp = float(world.temperature[cy, cx])
        humidity = float(world.humidity[cy, cx])
        light = float(world.light[cy, cx])
        terrain_speed = float(world.movement_speed[cy, cx])
        terrain_energy_cost = float(world.energy_cost[cy, cx])
        dist_home = math.hypot(self.x - colony.nest_x, self.y - colony.nest_y)
        self.distance_from_nest_max = max(self.distance_from_nest_max, dist_home)

        inputs = self._build_inputs(
            engine=engine,
            cx=cx,
            cy=cy,
            target_resource=target_resource,
            demand_here=demand_here,
            traffic_here=traffic_here,
            resource_here=resource_here,
            visibility=visibility,
            terrain_speed=terrain_speed,
            terrain_energy_cost=terrain_energy_cost,
            temp=temp,
            humidity=humidity,
            light=light,
            dist_home=dist_home,
            demand_ratio=colony.demand_ratio(),
        )
        outputs = self.brain.forward(inputs)
        self.last_inputs = inputs
        self.last_outputs = outputs

        self._try_pickup(engine, outputs[BrainOutput.PICKUP], target_resource)
        self._try_drop(engine, outputs[BrainOutput.DROP])
        self._lay_pheromones(engine, outputs, target_resource)
        self._apply_local_actions(engine, outputs)
        self._move(engine, cfg, outputs, rng, target_resource, terrain_speed, traffic_here, visibility)
        self._update_memory(outputs)

        self.energy = clamp(
            self.energy
            - cfg.energy_drain
            * terrain_energy_cost
            * (1.35 if self.has_cargo else 1.0)
            / max(0.2, self.genes.energy_efficiency),
            0.0,
            1.0,
        )
        if colony.is_storage_cell(self.x, self.y):
            self.energy = min(1.0, self.energy + cfg.storage_recharge)

        if int(self.age) % 180 == 0:
            self.inferred_role = infer_role(self)

    def _build_inputs(
        self,
        *,
        engine: "SimulationEngine",
        cx: int,
        cy: int,
        target_resource: ResourceType,
        demand_here: float,
        traffic_here: float,
        resource_here: float,
        visibility: float,
        terrain_speed: float,
        terrain_energy_cost: float,
        temp: float,
        humidity: float,
        light: float,
        dist_home: float,
        demand_ratio: float,
    ) -> np.ndarray:
        values = np.empty(INPUT_SIZE, dtype=np.float32)
        resource_layers = engine.resources.layers
        pheromone_layers = engine.pheromones.layers
        scent_radius = max(1, int(round(engine.config.ants.resource_scent_radius * visibility)))

        def amount_at_cell(resource_type: ResourceType) -> float:
            layer = resource_layers.get(resource_type)
            if layer is None:
                return 0.0
            return float(layer[cy, cx])

        def pheromone_at_cell(pheromone_type: PheromoneType) -> float:
            layer = pheromone_layers.get(pheromone_type)
            if layer is None:
                return 0.0
            return float(layer[cy, cx])

        home_angle = angle_to(self.x, self.y, engine.colony.nest_x, engine.colony.nest_y)
        wall_proximity = self._wall_proximity(engine, cx, cy)
        chamber_demand, storage_fullness = self._local_chamber_signals(engine)
        construction_amount = sum(
            amount_at_cell(resource_type)
            for resource_type in (ResourceType.WOOD_FIBER, ResourceType.MINERALS, ResourceType.SOIL, ResourceType.RESIN)
        )
        construction_scent = construction_amount
        protein_amount = (
            amount_at_cell(ResourceType.PROTEIN)
            + amount_at_cell(ResourceType.DEAD_INSECTS)
            + amount_at_cell(ResourceType.DEAD_ANTS)
        )

        values[BrainInput.ENERGY] = self.energy
        values[BrainInput.HEALTH] = self.health
        values[BrainInput.AGE] = min(1.0, self.age / max(1.0, self.lifespan))
        values[BrainInput.HAS_CARGO] = 1.0 if self.has_cargo else 0.0
        values[BrainInput.INVENTORY_AMOUNT] = min(1.0, self.inventory_amount / max(1e-6, self.inventory_capacity))
        values[BrainInput.INVENTORY_TYPE] = self._inventory_type_value()
        values[BrainInput.INVENTORY_CAPACITY] = min(1.0, self.inventory_capacity / 2.0)
        values[BrainInput.AGGRESSION] = min(1.0, self.aggression)
        values[BrainInput.CURIOSITY] = min(1.0, self.curiosity / 1.5)
        values[BrainInput.FEAR] = min(1.0, self.fear)
        values[BrainInput.STRENGTH] = min(1.0, self.strength / 1.5)
        values[BrainInput.SPEED] = min(1.0, self.speed / 1.5)
        values[BrainInput.VISION_RANGE] = min(1.0, self.vision_range / 1.6)
        values[BrainInput.PHEROMONE_SENSITIVITY] = min(1.0, self.pheromone_sensitivity / 1.8)
        values[BrainInput.TRAFFIC_TOLERANCE] = min(1.0, self.traffic_tolerance / 1.8)
        values[BrainInput.TERRAIN_SPEED] = terrain_speed
        values[BrainInput.TERRAIN_ENERGY_COST] = min(1.0, terrain_energy_cost / 2.5)
        values[BrainInput.TEMPERATURE] = temp
        values[BrainInput.HUMIDITY] = humidity
        values[BrainInput.LIGHT] = light
        values[BrainInput.FOOD_SCENT] = min(1.0, engine.resources.scent_at(ResourceType.LEAVES, self.x, self.y, scent_radius) / 10.0)
        values[BrainInput.WATER_SCENT] = min(1.0, engine.resources.scent_at(ResourceType.WATER, self.x, self.y, scent_radius) / 10.0)
        values[BrainInput.PROTEIN_SCENT] = min(1.0, engine.resources.scent_at(ResourceType.PROTEIN, self.x, self.y, scent_radius) / 10.0)
        values[BrainInput.CONSTRUCTION_SCENT] = min(1.0, construction_scent / 10.0)
        values[BrainInput.WASTE_SCENT] = min(1.0, engine.resources.scent_at(ResourceType.WASTE, self.x, self.y, scent_radius) / 10.0)
        values[BrainInput.TARGET_RESOURCE_AMOUNT] = min(1.0, resource_here / 8.0)
        values[BrainInput.FOOD_AMOUNT] = min(1.0, amount_at_cell(ResourceType.LEAVES) / 8.0)
        values[BrainInput.WATER_AMOUNT] = min(1.0, amount_at_cell(ResourceType.WATER) / 8.0)
        values[BrainInput.PROTEIN_AMOUNT] = min(1.0, protein_amount / 8.0)
        values[BrainInput.CONSTRUCTION_AMOUNT] = min(1.0, construction_amount / 8.0)
        values[BrainInput.WASTE_AMOUNT] = min(1.0, amount_at_cell(ResourceType.WASTE) / 8.0)
        values[BrainInput.FOOD_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.FOOD) / 90.0)
        values[BrainInput.WATER_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.WATER) / 80.0)
        values[BrainInput.PROTEIN_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.PROTEIN) / 80.0)
        values[BrainInput.CONSTRUCTION_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.CONSTRUCTION) / 65.0)
        values[BrainInput.DANGER_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.DANGER) / 100.0)
        values[BrainInput.RECRUITMENT_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.RECRUITMENT) / 90.0)
        values[BrainInput.TRAFFIC_PHEROMONE] = min(1.0, traffic_here / 65.0)
        values[BrainInput.TERRITORY_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.TERRITORY) / 70.0)
        values[BrainInput.WASTE_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.WASTE) / 70.0)
        values[BrainInput.QUEEN_SIGNAL] = min(1.0, engine.colony.storage_signal_at(self.x, self.y))
        values[BrainInput.EMERGENCY_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.EMERGENCY) / 100.0)
        values[BrainInput.DEMAND_PHEROMONE] = min(1.0, demand_here / 110.0)
        values[BrainInput.STORAGE_FULL_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.STORAGE_FULL) / 70.0)
        values[BrainInput.NEST_ENTRANCE_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.NEST_ENTRANCE) / 90.0)
        values[BrainInput.DEATH_PHEROMONE] = min(1.0, pheromone_at_cell(PheromoneType.DEATH) / 80.0)
        values[BrainInput.ALLY_DENSITY] = self._ally_density(engine, cx, cy)
        values[BrainInput.ENEMY_DENSITY] = 0.0
        values[BrainInput.PREDATOR_SIGNAL] = 0.0
        values[BrainInput.TRAFFIC_CONGESTION] = min(1.0, traffic_here / 75.0)
        values[BrainInput.HOME_DISTANCE] = min(1.0, dist_home / 120.0)
        values[BrainInput.HOME_DIRECTION_X] = 0.5 + 0.5 * math.cos(home_angle)
        values[BrainInput.HOME_DIRECTION_Y] = 0.5 + 0.5 * math.sin(home_angle)
        values[BrainInput.WALL_PROXIMITY] = wall_proximity
        values[BrainInput.CHAMBER_DEMAND] = chamber_demand
        values[BrainInput.STORAGE_FULLNESS] = storage_fullness
        values[BrainInput.TARGET_PRIORITY] = min(1.0, engine.colony.resource_priority(target_resource))
        values[BrainInput.TARGET_SATURATION] = min(1.0, engine.colony.resource_saturation(target_resource))
        values[BrainInput.MEMORY_0 : BrainInput.MEMORY_3 + 1] = self.memory
        values[BrainInput.ENERGY_STRESS] = 1.0 - self.energy
        values[BrainInput.RANDOM_DRIFT] = ((self.ant_id * 17 + int(self.age) * 31) % 997) / 997.0
        return values

    def _pheromone(self, engine: "SimulationEngine", pheromone_type: PheromoneType) -> float:
        layer = engine.pheromones.layers.get(pheromone_type)
        if layer is None:
            return 0.0
        return engine.pheromones.sample(pheromone_type, self.x, self.y)

    def _resource_scent(self, engine: "SimulationEngine", resource_type: ResourceType) -> float:
        if resource_type not in engine.resources.layers:
            return 0.0
        radius = max(1, int(round(engine.config.ants.resource_scent_radius * engine.world.visibility_at(self.x, self.y))))
        return engine.resources.scent_at(resource_type, self.x, self.y, radius)

    def _construction_resource_amount(self, engine: "SimulationEngine") -> float:
        return sum(
            engine.resources.amount_at(resource_type, self.x, self.y)
            for resource_type in (ResourceType.WOOD_FIBER, ResourceType.MINERALS, ResourceType.SOIL, ResourceType.RESIN)
        )

    def _ally_density(self, engine: "SimulationEngine", cx: int | None = None, cy: int | None = None) -> float:
        if cx is None or cy is None:
            cx, cy = engine.world.cell(self.x, self.y)
        x0 = max(0, cx - 1)
        x1 = min(engine.world.width, cx + 2)
        y0 = max(0, cy - 1)
        y1 = min(engine.world.height, cy + 2)
        integral = engine.ant_density_integral
        if int(integral[-1, -1]) > 0:
            nearby = (
                int(integral[y1, x1])
                - int(integral[y0, x1])
                - int(integral[y1, x0])
                + int(integral[y0, x0])
            )
        else:
            nearby = int(np.sum(engine.ant_density[y0:y1, x0:x1]))
        if nearby > 0:
            nearby -= 1
        return min(1.0, nearby / 18.0)

    def _wall_proximity(self, engine: "SimulationEngine", cx: int, cy: int) -> float:
        obstacles = engine.world.obstacles
        blocked = int(cx <= 0 or obstacles[cy, cx - 1])
        blocked += int(cx >= engine.world.width - 1 or obstacles[cy, cx + 1])
        blocked += int(cy <= 0 or obstacles[cy - 1, cx])
        blocked += int(cy >= engine.world.height - 1 or obstacles[cy + 1, cx])
        return blocked / 4.0

    def _local_chamber_signals(self, engine: "SimulationEngine") -> tuple[float, float]:
        if (self.x - engine.colony.nest_x) ** 2 + (self.y - engine.colony.nest_y) ** 2 > (engine.colony.storage_radius + 18.0) ** 2:
            return 0.0, 0.0
        chamber = engine.colony.chamber_at(self.x, self.y)
        if chamber is None:
            return 0.0, 0.0
        best_demand = 0.0
        best_fullness = 0.0
        for resource_type in chamber.stored_resources():
            desired = max(1.0, chamber.desired_for(resource_type))
            best_demand = max(best_demand, min(1.0, chamber.demand(resource_type) / desired))
            best_fullness = max(best_fullness, chamber.fullness(resource_type))
        return best_demand, best_fullness

    def _inventory_type_value(self) -> float:
        if self.inventory_type is None:
            return 0.0
        return RESOURCE_TYPE_INPUT_VALUES[self.inventory_type]

    def _try_pickup(self, engine: "SimulationEngine", pickup_signal: float, target_resource: ResourceType | None = None) -> None:
        if self.has_cargo:
            return
        chamber_zone = engine.colony.storage_radius + 13.0
        near_chamber_cluster = (self.x - engine.colony.nest_x) ** 2 + (self.y - engine.colony.nest_y) ** 2 <= chamber_zone * chamber_zone
        internal_pickup_tick = int(self.age + self.ant_id) % 3 == 0
        if pickup_signal >= 0.18 and internal_pickup_tick and near_chamber_cluster and self._try_chamber_pickup(engine):
            return
        if pickup_signal < 0.35:
            return
        resource_type = self._best_pickup_resource(engine) or target_resource or self.target_resource
        amount_here = engine.resources.amount_at(resource_type, self.x, self.y)
        if amount_here <= 0.15:
            resource_type = self._best_pickup_resource(engine)
            if resource_type is None:
                return
            amount_here = engine.resources.amount_at(resource_type, self.x, self.y)
        if resource_type is None:
            return
        if amount_here <= 0.15:
            return
        capacity = engine.config.ants.inventory_capacity * self.inventory_capacity
        taken = engine.resources.take_at(resource_type, self.x, self.y, min(capacity, engine.config.ants.pickup_amount))
        if taken > 0:
            self.inventory_type = resource_type
            self.inventory_amount = taken
            self.pickup_chamber_id = None
            self.cargo_source = "world"
            self.cargo_destination_kind = None
            self.last_pickup_tick = engine.tick
            engine.pheromones.add_at(RESOURCE_PHEROMONES[resource_type], self.x, self.y, 16.0, radius=1)

    def _try_chamber_pickup(self, engine: "SimulationEngine") -> bool:
        chamber = engine.colony.chamber_at(self.x, self.y)
        if chamber is None:
            return False
        candidate = self._best_chamber_pickup(engine, chamber)
        if candidate is None:
            return False
        resource_type, score, destination = candidate
        if score <= 0.0:
            return False
        capacity = engine.config.ants.inventory_capacity * self.inventory_capacity
        taken = chamber.remove_resource(resource_type, min(capacity, engine.config.ants.pickup_amount))
        if taken <= 0.0:
            return False
        self.inventory_type = resource_type
        self.inventory_amount = taken
        self.pickup_chamber_id = chamber.chamber_id
        if chamber.is_fungus_farm:
            self.cargo_source = "farm"
        elif chamber.is_processor:
            self.cargo_source = "processor"
        else:
            self.cargo_source = "storage"
        self.target_chamber_id = destination.chamber_id if destination is not None else None
        if destination is not None and destination.is_fungus_farm:
            self.cargo_destination_kind = "farm"
        elif destination is not None and destination.is_processor:
            self.cargo_destination_kind = "processor"
        elif destination is not None and destination.is_nursery:
            self.cargo_destination_kind = "nursery"
        else:
            self.cargo_destination_kind = "storage" if resource_type in PROCESSED_LOGISTICS_RESOURCES else "processor"
        self.target_resource = resource_type
        self.last_pickup_tick = engine.tick
        engine.colony.refresh_aggregate_inventory()
        engine.pheromones.add_at(PheromoneType.DEMAND, self.x, self.y, 3.5, radius=1)
        return True

    def _best_chamber_pickup(self, engine: "SimulationEngine", chamber) -> tuple[ResourceType, float, object | None] | None:
        best_resource: ResourceType | None = None
        best_destination = None
        best_score = 0.0
        if chamber.is_production:
            for resource_type in chamber.outputs:
                amount = chamber.inventory.get(resource_type)
                if amount <= 0.12:
                    continue
                destination = engine.colony.best_storage_for_output(resource_type, self.x, self.y)
                if destination is None:
                    continue
                fullness = chamber.output_fullness(resource_type)
                pressure = engine.colony.processor_output_pressure(resource_type)
                score = 7.5 * fullness + 4.0 * pressure + amount
                if score > best_score:
                    best_resource = resource_type
                    best_destination = destination
                    best_score = score
        elif chamber.is_storage:
            for resource_type, amount in chamber.inventory.amounts.items():
                if amount <= 0.12:
                    continue
                destination = engine.colony.best_processor_for_input(resource_type, self.x, self.y)
                if destination is None:
                    continue
                processor_demand = engine.colony.processor_input_demand(resource_type)
                if processor_demand <= 0.02:
                    continue
                storage_priority = engine.colony.resource_priority(resource_type)
                reserve_penalty = 0.55 if storage_priority > 0.75 and chamber.fullness(resource_type) < 0.35 else 1.0
                score = (6.5 * processor_demand + min(3.0, amount * 0.08)) * reserve_penalty
                if score > best_score:
                    best_resource = resource_type
                    best_destination = destination
                    best_score = score
        if best_resource is None:
            return None
        return best_resource, best_score, best_destination

    def _try_drop(self, engine: "SimulationEngine", drop_signal: float) -> None:
        if not self.has_cargo or drop_signal < 0.28:
            return
        if self.inventory_type is None:
            return
        chamber_zone = engine.colony.storage_radius + 13.0
        if (self.x - engine.colony.nest_x) ** 2 + (self.y - engine.colony.nest_y) ** 2 > chamber_zone * chamber_zone:
            return
        chamber = engine.colony.chamber_at(self.x, self.y)
        target_chamber = engine.colony.chamber_by_id(self.target_chamber_id)
        target_is_valid = target_chamber is not None and target_chamber.can_accept_input(self.inventory_type)
        if target_is_valid:
            arrival_radius = target_chamber.radius + 1.5
            if (self.x - target_chamber.x) ** 2 + (self.y - target_chamber.y) ** 2 <= arrival_radius * arrival_radius:
                chamber = target_chamber
            can_drop_in_chamber = chamber is target_chamber
        else:
            can_drop_in_chamber = chamber is not None and chamber.can_accept_input(self.inventory_type)
        has_storage_capacity = True
        if not can_drop_in_chamber:
            has_storage_capacity = engine.colony.available_storage_capacity(self.inventory_type) > 1e-6
        can_drop_overflow = not has_storage_capacity and engine.colony.is_overflow_cell(self.x, self.y)
        if not can_drop_in_chamber and not can_drop_overflow:
            return
        amount = self.inventory_amount
        accepted = engine.colony.add_to_storage(self.inventory_type, amount, engine.tick, self.x, self.y)
        if accepted <= 0.0:
            return
        self.deliveries += 1
        self.resources_delivered += accepted
        self.resources_delivered_by_type[self.inventory_type] = self.resources_delivered_by_type.get(self.inventory_type, 0.0) + accepted
        self.last_delivery_ticks = max(1, engine.tick - self.last_pickup_tick)
        self.inventory_type = None
        self.inventory_amount = 0.0
        self.target_chamber_id = None
        self.pickup_chamber_id = None
        self.cargo_source = None
        self.cargo_destination_kind = None
        self.target_resource = self._target_resource(engine, engine.config.ants)
        engine.pheromones.add_at(PheromoneType.DEMAND, self.x, self.y, 4.0, radius=1)

    def _lay_pheromones(self, engine: "SimulationEngine", outputs: np.ndarray, target_resource: ResourceType) -> None:
        resource_output = {
            PheromoneType.FOOD: BrainOutput.LAY_FOOD,
            PheromoneType.WATER: BrainOutput.LAY_WATER,
            PheromoneType.PROTEIN: BrainOutput.LAY_PROTEIN,
            PheromoneType.CONSTRUCTION: BrainOutput.LAY_CONSTRUCTION,
            PheromoneType.WASTE: BrainOutput.LAY_WASTE,
        }
        if self.has_cargo:
            pheromone_type = RESOURCE_PHEROMONES.get(self.inventory_type)
            if pheromone_type is not None:
                signal = float(outputs[resource_output.get(pheromone_type, BrainOutput.LAY_FOOD)])
                amount = 1.0 + 7.0 * signal
                engine.pheromones.add_at(pheromone_type, self.x, self.y, amount)
            else:
                amount = 1.0 + 7.0 * float(outputs[BrainOutput.LAY_DEMAND])
                engine.pheromones.add_at(PheromoneType.DEMAND, self.x, self.y, amount * 0.32)
        else:
            if target_resource not in GATHERABLE_RESOURCES:
                resource_type = None
            else:
                resource_type: ResourceType | None = target_resource
            if resource_type is not None and engine.colony.resource_priority(target_resource) < 0.05:
                resource_type = None
            elif resource_type is not None and engine.resources.amount_at(target_resource, self.x, self.y) <= 0.2:
                resource_type = None
                if int(self.age + self.ant_id) % 15 == 0:
                    resource_type = self._best_pickup_resource(engine, require_signal=False)
            if resource_type is not None:
                pheromone_type = RESOURCE_PHEROMONES[resource_type]
                signal = float(outputs[resource_output.get(pheromone_type, BrainOutput.LAY_FOOD)])
                if signal > 0.48:
                    engine.pheromones.add_at(pheromone_type, self.x, self.y, 1.0 + 5.0 * signal)

        for output_index, pheromone_type in (
            (BrainOutput.LAY_DANGER, PheromoneType.DANGER),
            (BrainOutput.LAY_RECRUITMENT, PheromoneType.RECRUITMENT),
            (BrainOutput.LAY_TERRITORY, PheromoneType.TERRITORY),
            (BrainOutput.LAY_DEMAND, PheromoneType.DEMAND),
            (BrainOutput.LAY_EMERGENCY, PheromoneType.EMERGENCY),
        ):
            signal = float(outputs[output_index])
            threshold = 0.70 if pheromone_type == PheromoneType.DEMAND else 0.82
            if signal <= threshold:
                continue
            amount = 0.6 + 3.2 * signal
            engine.pheromones.add_at(pheromone_type, self.x, self.y, amount)
            if pheromone_type == PheromoneType.DANGER:
                self.danger_pheromone_laid += amount
            elif pheromone_type == PheromoneType.RECRUITMENT:
                self.recruitment_pheromone_laid += amount

        traffic_amount = 0.8 + 2.8 * float(outputs[BrainOutput.LAY_TRAFFIC])
        engine.pheromones.add_at(PheromoneType.TRAFFIC, self.x, self.y, traffic_amount)
        if engine.colony.territory is not None and int(self.age) % 12 == 0:
            engine.colony.territory.mark(self.x, self.y, 0.02)

    def _apply_local_actions(self, engine: "SimulationEngine", outputs: np.ndarray) -> None:
        if float(outputs[BrainOutput.DIG]) > 0.68:
            self.digging_ticks += 1
        chamber = None
        if float(outputs[BrainOutput.BUILD_REINFORCE]) > 0.68:
            self.building_ticks += 1
            chamber = engine.colony.chamber_at(self.x, self.y)
            if chamber is not None and chamber.health < 1.0:
                repaired = min(0.002 * self.strength, 1.0 - chamber.health)
                chamber.health += repaired
                self.construction_contributed += repaired
        if float(outputs[BrainOutput.ATTACK]) > 0.70:
            self.attack_intent_ticks += 1
        if float(outputs[BrainOutput.DEFEND]) > 0.62 and engine.colony.storage_signal_at(self.x, self.y) > 0.45:
            self.defend_ticks += 1
        if float(outputs[BrainOutput.GROOM_HELP]) > 0.62 and self._ally_density(engine) > 0.04:
            self.grooming_ticks += 1
            self.allies_helped += 0.01

        tend_larvae = float(outputs[BrainOutput.TEND_LARVAE])
        tend_fungus = float(outputs[BrainOutput.TEND_FUNGUS])
        if chamber is None and (tend_larvae > 0.55 or tend_fungus > 0.55):
            chamber = engine.colony.chamber_at(self.x, self.y)
        if chamber is not None and chamber.is_nursery and tend_larvae > 0.55:
            self.nursing_ticks += 1
            if chamber.brood_state is not None and chamber.brood_state.total_brood > 0:
                self.larvae_helped += 0.01
        if chamber is not None and chamber.is_fungus_farm and tend_fungus > 0.55:
            self.fungus_tended += 0.01

    def _update_memory(self, outputs: np.ndarray) -> None:
        updated = np.array(
            [
                outputs[BrainOutput.MEMORY_0],
                outputs[BrainOutput.MEMORY_1],
                outputs[BrainOutput.MEMORY_2],
                outputs[BrainOutput.MEMORY_3],
            ],
            dtype=np.float32,
        )
        self.memory = np.clip(self.memory * 0.80 + updated * 0.20, -1.0, 1.0).astype(np.float32)

    def _move(
        self,
        engine: "SimulationEngine",
        cfg: AntConfig,
        outputs: np.ndarray,
        rng: "Generator",
        target_resource: ResourceType,
        terrain_speed: float,
        traffic_here: float,
        visibility: float,
    ) -> None:
        world = engine.world
        colony = engine.colony

        desired_angle = self._desired_angle(engine, cfg, outputs, rng, target_resource, visibility)
        steer = angle_delta(desired_angle, self.angle)
        neural_turn = float(outputs[BrainOutput.TURN]) * cfg.max_turn
        self.angle = wrap_angle(self.angle + clamp(steer, -cfg.max_turn, cfg.max_turn) * 0.75 + neural_turn * 0.25)

        traffic_penalty = clamp(traffic_here / 75.0, 0.0, 0.70)
        tolerance = clamp(self.genes.traffic_tolerance, 0.2, 1.8)
        neural_speed = 0.45 + 0.55 * max(0.0, float(outputs[BrainOutput.MOVE_SPEED]))
        cargo_penalty = self._cargo_speed_penalty()
        low_energy_penalty = 0.45 + 0.55 * self.energy
        speed = cfg.base_speed * self.genes.speed * terrain_speed * neural_speed * cargo_penalty * low_energy_penalty
        avoid_congestion = float(outputs[BrainOutput.AVOID_CONGESTION])
        speed *= 1.0 - (traffic_penalty * avoid_congestion) / tolerance
        speed = max(0.025, speed)

        nx = self.x + math.cos(self.angle) * speed
        ny = self.y + math.sin(self.angle) * speed
        if world.is_blocked(nx, ny):
            self.angle = wrap_angle(self.angle + float(rng.uniform(1.6, 3.2)))
            nx = self.x + math.cos(self.angle) * speed * 0.35
            ny = self.y + math.sin(self.angle) * speed * 0.35
            if world.is_blocked(nx, ny):
                self.velocity = 0.0
                self.velocity_x = 0.0
                self.velocity_y = 0.0
                return

        traveled = math.hypot(nx - self.x, ny - self.y)
        self.velocity = traveled
        self.velocity_x = nx - self.x
        self.velocity_y = ny - self.y
        self.x = clamp(nx, 0.0, world.width - 0.001)
        self.y = clamp(ny, 0.0, world.height - 0.001)
        self.distance_traveled += traveled
        colony.stats.distance_traveled += traveled

    def _desired_angle(
        self,
        engine: "SimulationEngine",
        cfg: AntConfig,
        outputs: np.ndarray,
        rng: "Generator",
        target_resource: ResourceType,
        visibility: float,
    ) -> float:
        sensor_distance = cfg.pheromone_sensor_distance * max(0.5, self.genes.vision_range) * max(0.35, visibility)
        left_angle = self.angle - 0.75
        right_angle = self.angle + 0.75
        forward_angle = self.angle

        if self.has_cargo:
            field_type = PheromoneType.DEMAND
            left = engine.pheromones.sample_direction(field_type, self.x, self.y, left_angle, sensor_distance)
            right = engine.pheromones.sample_direction(field_type, self.x, self.y, right_angle, sensor_distance)
            forward = engine.pheromones.sample_direction(field_type, self.x, self.y, forward_angle, sensor_distance)
            target_x, target_y = self._drop_target(engine)
            home_angle = angle_to(self.x, self.y, target_x, target_y)
            gradient_turn = clamp((right - left) / 45.0, -1.0, 1.0)
            if max(left, right, forward) > 2.0:
                pheromone_angle = wrap_angle(self.angle + gradient_turn * 0.95)
                blend = 0.55
                return self._blend_angles(home_angle, pheromone_angle, blend)
            return home_angle + float(rng.normal(0.0, 0.10))

        if target_resource not in GATHERABLE_RESOURCES:
            return wrap_angle(self.angle + float(rng.normal(0.0, 0.22)))

        pheromone_type = RESOURCE_PHEROMONES[target_resource]
        priority = engine.colony.resource_priority(target_resource)
        weight = RESOURCE_TRAIL_WEIGHTS[target_resource] * (0.26 + 1.70 * priority) * self.genes.pheromone_sensitivity
        scent_weight = 3.4 * weight
        resource_left = weight * engine.pheromones.sample_direction(pheromone_type, self.x, self.y, left_angle, sensor_distance)
        resource_right = weight * engine.pheromones.sample_direction(pheromone_type, self.x, self.y, right_angle, sensor_distance)
        resource_forward = weight * engine.pheromones.sample_direction(
            pheromone_type, self.x, self.y, forward_angle, sensor_distance
        )
        resource_left += scent_weight * engine.resources.scent_at(
            target_resource,
            self.x + math.cos(left_angle) * sensor_distance,
            self.y + math.sin(left_angle) * sensor_distance,
            cfg.resource_scent_radius,
        )
        resource_right += scent_weight * engine.resources.scent_at(
            target_resource,
            self.x + math.cos(right_angle) * sensor_distance,
            self.y + math.sin(right_angle) * sensor_distance,
            cfg.resource_scent_radius,
        )
        resource_forward += scent_weight * engine.resources.scent_at(
            target_resource,
            self.x + math.cos(forward_angle) * sensor_distance,
            self.y + math.sin(forward_angle) * sensor_distance,
            cfg.resource_scent_radius,
        )
        traffic_left = engine.pheromones.sample_direction(PheromoneType.TRAFFIC, self.x, self.y, left_angle, sensor_distance)
        traffic_right = engine.pheromones.sample_direction(PheromoneType.TRAFFIC, self.x, self.y, right_angle, sensor_distance)

        avoid_congestion = float(outputs[BrainOutput.AVOID_CONGESTION])
        left = resource_left - traffic_left * 0.16 * avoid_congestion
        right = resource_right - traffic_right * 0.16 * avoid_congestion
        forward = resource_forward
        gradient_turn = clamp((right - left) / 32.0, -1.0, 1.0)
        if max(left, right, forward) > 0.8:
            return wrap_angle(self.angle + gradient_turn * 0.95)

        explore = (float(outputs[BrainOutput.EXPLORE]) - 0.5) * self.genes.curiosity
        return wrap_angle(self.angle + explore * 0.40 + float(rng.normal(0.0, 0.22)))

    def _target_resource(self, engine: "SimulationEngine", cfg: AntConfig) -> ResourceType:
        best_resource = ResourceType.LEAVES
        best_score = -1.0
        for resource_type in GATHERABLE_RESOURCES:
            score = self._resource_score(engine, cfg, resource_type)
            if score > best_score:
                best_resource = resource_type
                best_score = score
        return best_resource

    def _best_pickup_resource(self, engine: "SimulationEngine", require_signal: bool = True) -> ResourceType | None:
        best_resource: ResourceType | None = None
        best_score = 0.0
        for resource_type in GATHERABLE_RESOURCES:
            amount = engine.resources.amount_at(resource_type, self.x, self.y)
            if amount <= 0.15:
                continue
            priority = engine.colony.resource_priority(resource_type)
            pheromone = engine.pheromones.sample(RESOURCE_PHEROMONES[resource_type], self.x, self.y)
            saturation_penalty = 1.0 / (1.0 + engine.colony.resource_saturation(resource_type) * 8.0)
            saturated_floor = 0.12 if engine.colony.resource_demand_state(resource_type).is_saturated else 1.0
            score = amount * 0.75 * saturated_floor + priority * 12.0 + pheromone * 0.035
            score *= saturation_penalty
            score *= RESOURCE_TRAIL_WEIGHTS[resource_type]
            if score > best_score:
                best_resource = resource_type
                best_score = score
        if best_resource is None:
            return None
        if require_signal and best_score < 0.12:
            return None
        return best_resource

    def _resource_score(self, engine: "SimulationEngine", cfg: AntConfig, resource_type: ResourceType) -> float:
        amount = engine.resources.amount_at(resource_type, self.x, self.y)
        scent = engine.resources.scent_at(resource_type, self.x, self.y, cfg.resource_scent_radius)
        pheromone = engine.pheromones.sample(RESOURCE_PHEROMONES[resource_type], self.x, self.y)
        priority = engine.colony.resource_priority(resource_type)
        saturation = engine.colony.resource_saturation(resource_type)
        traffic = engine.pheromones.sample(PheromoneType.TRAFFIC, self.x, self.y)
        local_bonus = 2.5 if amount > 0.15 else 0.0
        resource_index = GATHERABLE_RESOURCES.index(resource_type)
        curiosity_bias = self.genes.curiosity * 0.08 * ((self.ant_id + resource_index * 7) % 11) / 10.0
        saturation_penalty = 1.0 / (1.0 + saturation * 8.0)
        traffic_penalty = max(0.0, 1.0 - traffic / 170.0)
        score = (
            local_bonus * (0.30 + priority)
            + amount * 0.38 * (0.25 + priority)
            + scent * 1.45 * self.genes.pheromone_sensitivity * (0.20 + priority)
            + pheromone * 0.035 * (0.35 + priority)
            + priority * 5.8
            + curiosity_bias
        )
        return score * RESOURCE_TRAIL_WEIGHTS[resource_type] * saturation_penalty * traffic_penalty

    def _cargo_speed_penalty(self) -> float:
        if not self.has_cargo or self.inventory_type is None:
            return 1.0
        weight = RESOURCE_SPECS[self.inventory_type].weight
        return clamp(0.82 / max(0.7, weight), 0.58, 0.86)

    def _drop_target(self, engine: "SimulationEngine") -> tuple[float, float]:
        if self.inventory_type is None:
            self.target_chamber_id = None
            return engine.colony.nest_x, engine.colony.nest_y
        current_target = engine.colony.chamber_by_id(self.target_chamber_id)
        if current_target is not None and current_target.can_accept_input(self.inventory_type):
            return current_target.x, current_target.y
        if self.inventory_type in PROCESSED_LOGISTICS_RESOURCES:
            if engine.colony.internal_input_demand(self.inventory_type) > 0.05:
                chamber = engine.colony.best_drop_chamber(self.inventory_type, self.x, self.y)
                if chamber is not None and chamber.is_nursery:
                    self.target_chamber_id = chamber.chamber_id
                    self.cargo_destination_kind = "nursery"
                    return chamber.x, chamber.y
            chamber = engine.colony.best_storage_for_output(self.inventory_type, self.x, self.y)
            if chamber is not None:
                self.target_chamber_id = chamber.chamber_id
                self.cargo_destination_kind = "storage"
                return chamber.x, chamber.y
        chamber = engine.colony.best_drop_chamber(self.inventory_type, self.x, self.y)
        if chamber is None:
            chamber = engine.colony.nearest_chamber_accepting(self.inventory_type, self.x, self.y)
        if chamber is None:
            self.target_chamber_id = None
            return engine.colony.nest_x, engine.colony.nest_y
        self.target_chamber_id = chamber.chamber_id
        if chamber.is_fungus_farm:
            self.cargo_destination_kind = "farm"
        elif chamber.is_processor:
            self.cargo_destination_kind = "processor"
        elif chamber.is_nursery:
            self.cargo_destination_kind = "nursery"
        else:
            self.cargo_destination_kind = "storage"
        return chamber.x, chamber.y

    def _blend_angles(self, a: float, b: float, b_weight: float) -> float:
        ax, ay = math.cos(a), math.sin(a)
        bx, by = math.cos(b), math.sin(b)
        x = ax * (1.0 - b_weight) + bx * b_weight
        y = ay * (1.0 - b_weight) + by * b_weight
        return math.atan2(y, x)
