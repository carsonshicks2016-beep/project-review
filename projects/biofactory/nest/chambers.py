from __future__ import annotations

from dataclasses import dataclass, field

from biofactory.resources.inventory import Inventory
from biofactory.resources.resource_types import ResourceType


@dataclass
class Chamber:
    chamber_id: int
    chamber_type: str
    x: float
    y: float
    radius: float
    accepted_inputs: set[ResourceType]
    outputs: set[ResourceType] = field(default_factory=set)
    inventory: Inventory = field(default_factory=Inventory)
    desired_inventory_by_resource: dict[ResourceType, float] = field(default_factory=dict)
    capacity_by_resource: dict[ResourceType, float] = field(default_factory=dict)
    processing_state: object | None = None
    fungus_farm_state: object | None = None
    brood_state: object | None = None
    health: float = 1.0

    @property
    def primary_resource(self) -> ResourceType | None:
        if self.outputs:
            return next(iter(self.outputs))
        if self.accepted_inputs:
            return next(iter(self.accepted_inputs))
        return None

    @property
    def is_processor(self) -> bool:
        return self.processing_state is not None

    @property
    def is_fungus_farm(self) -> bool:
        return self.fungus_farm_state is not None

    @property
    def is_nursery(self) -> bool:
        return self.brood_state is not None

    @property
    def is_production(self) -> bool:
        return self.processing_state is not None or self.fungus_farm_state is not None

    @property
    def is_input_consumer(self) -> bool:
        return self.is_production or self.is_nursery

    @property
    def is_storage(self) -> bool:
        return not self.is_input_consumer

    def stored_resources(self) -> set[ResourceType]:
        return set(self.accepted_inputs) | set(self.outputs) | set(self.capacity_by_resource)

    def contains(self, x: float, y: float) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius * self.radius

    def accepts(self, resource_type: ResourceType) -> bool:
        return resource_type in self.accepted_inputs

    def stores(self, resource_type: ResourceType) -> bool:
        return resource_type in self.accepted_inputs or resource_type in self.outputs or resource_type in self.capacity_by_resource

    def desired_for(self, resource_type: ResourceType) -> float:
        if not self.stores(resource_type):
            return 0.0
        return self.desired_inventory_by_resource.get(resource_type, 0.0)

    def capacity_for(self, resource_type: ResourceType) -> float:
        if not self.stores(resource_type):
            return 0.0
        return self.capacity_by_resource.get(resource_type, self.desired_for(resource_type))

    def fullness(self, resource_type: ResourceType) -> float:
        capacity = self.capacity_for(resource_type)
        if capacity <= 0.0:
            return 1.0
        return max(0.0, min(1.0, self.inventory.get(resource_type) / capacity))

    def available_capacity(self, resource_type: ResourceType) -> float:
        return max(0.0, self.capacity_for(resource_type) - self.inventory.get(resource_type))

    def can_accept(self, resource_type: ResourceType) -> bool:
        return self.can_accept_input(resource_type)

    def can_accept_input(self, resource_type: ResourceType) -> bool:
        return self.accepts(resource_type) and self.available_capacity(resource_type) > 1e-6

    def can_release_output(self, resource_type: ResourceType) -> bool:
        return resource_type in self.outputs and self.inventory.get(resource_type) > 1e-6

    def add_resource(self, resource_type: ResourceType, amount: float) -> float:
        if amount <= 0.0 or not self.stores(resource_type):
            return 0.0
        accepted = min(amount, self.available_capacity(resource_type))
        self.inventory.add(resource_type, accepted)
        return accepted

    def remove_resource(self, resource_type: ResourceType, amount: float) -> float:
        if amount <= 0.0 or not self.stores(resource_type):
            return 0.0
        return self.inventory.remove(resource_type, amount)

    def demand(self, resource_type: ResourceType) -> float:
        if self.is_input_consumer and resource_type in self.accepted_inputs:
            return self.input_demand(resource_type)
        if not self.accepts(resource_type):
            return 0.0
        return max(0.0, self.desired_for(resource_type) - self.inventory.get(resource_type))

    def input_demand(self, resource_type: ResourceType) -> float:
        if resource_type not in self.accepted_inputs:
            return 0.0
        if self.is_nursery:
            return self._nursery_input_demand(resource_type)
        if self.is_fungus_farm:
            return self._fungus_farm_input_demand(resource_type)
        if not self.is_processor or self.processing_state is None:
            return 0.0
        recipe = self.processing_state.recipe
        if resource_type not in recipe.inputs:
            return 0.0
        for output_type in recipe.outputs:
            if self.output_fullness(output_type) >= 1.0:
                return 0.0
        desired = self.desired_for(resource_type)
        if desired <= 0.0:
            desired = recipe.inputs[resource_type] * 2.0
        return max(0.0, desired - self.inventory.get(resource_type))

    def _fungus_farm_input_demand(self, resource_type: ResourceType) -> float:
        if self.fungus_farm_state is None:
            return 0.0
        if self.output_fullness(ResourceType.FUNGUS) >= 1.0:
            return 0.0
        leaves = self.inventory.get(ResourceType.LEAVES)
        water = self.inventory.get(ResourceType.WATER)
        if resource_type == ResourceType.COMPOST and (leaves <= 0.5 or water <= 0.25):
            return 0.0
        desired = self.desired_for(resource_type)
        if desired <= 0.0:
            return 0.0
        demand = max(0.0, desired - self.inventory.get(resource_type))
        if resource_type == ResourceType.COMPOST:
            demand *= 0.45
        return demand

    def _nursery_input_demand(self, resource_type: ResourceType) -> float:
        if self.brood_state is None:
            return 0.0
        desired = self.desired_for(resource_type)
        if desired <= 0.0:
            return 0.0
        demand = max(0.0, desired - self.inventory.get(resource_type))
        brood = self.brood_state.larvae + self.brood_state.pupae + max(0.25, self.brood_state.eggs * 0.35)
        if brood <= 0.0:
            demand *= 0.35
        if (
            (self.brood_state.blocked_reason == "missing_food" and resource_type == ResourceType.FOOD)
            or (self.brood_state.blocked_reason == "missing_water" and resource_type == ResourceType.WATER)
        ):
            demand *= 1.35
        return demand

    def output_fullness(self, resource_type: ResourceType) -> float:
        if resource_type not in self.outputs:
            return 1.0
        return self.fullness(resource_type)

    def process_tick(self, worker_count: int, config) -> bool:
        if self.processing_state is None:
            return False
        bonus = min(config.processor_max_worker_bonus, worker_count * config.processor_worker_bonus_per_ant)
        work_done = config.processor_base_work_per_tick + bonus
        self.processing_state.last_worker_count = worker_count
        return self.processing_state.advance(self, work_done)

    def farm_tick(self, worker_count: int, config, tick: int) -> float:
        if self.fungus_farm_state is None:
            return 0.0
        return self.fungus_farm_state.advance(self, worker_count, config, tick)

    def nursery_tick(self, worker_count: int, config, tick: int) -> int:
        if self.brood_state is None:
            return 0
        return self.brood_state.advance(self, worker_count, config, tick)
