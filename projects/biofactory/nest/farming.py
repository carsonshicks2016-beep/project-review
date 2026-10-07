from __future__ import annotations

from dataclasses import dataclass

from biofactory.resources.resource_types import ResourceType


@dataclass
class FungusFarmState:
    biomass: float = 0.0
    fungus_harvested: float = 0.0
    blocked_reason: str | None = "idle"
    last_worker_count: int = 0
    last_efficiency: float = 0.0
    last_compost_boost: float = 0.0
    last_harvest_tick: int = -999999

    def input_missing(self, chamber) -> dict[ResourceType, float]:
        missing: dict[ResourceType, float] = {}
        if chamber.inventory.get(ResourceType.LEAVES) <= 1e-6:
            missing[ResourceType.LEAVES] = 1.0
        if chamber.inventory.get(ResourceType.WATER) <= 1e-6:
            missing[ResourceType.WATER] = 1.0
        return missing

    def output_room_missing(self, chamber) -> dict[ResourceType, float]:
        if chamber.available_capacity(ResourceType.FUNGUS) <= 1e-6:
            return {ResourceType.FUNGUS: 1.0}
        return {}

    def can_grow(self, chamber) -> bool:
        if chamber.health <= 0.0:
            self.blocked_reason = "damaged"
            return False
        if self.output_room_missing(chamber):
            self.blocked_reason = "output_full"
            return False
        if self.input_missing(chamber):
            self.blocked_reason = "missing_inputs"
            return False
        return True

    def advance(self, chamber, worker_count: int, config, tick: int = 0) -> float:
        self.last_worker_count = worker_count
        self.last_compost_boost = 0.0
        if not self.can_grow(chamber):
            self.last_efficiency = 0.0
            return 0.0

        worker_bonus = min(config.fungus_farm_max_worker_bonus, worker_count * config.fungus_farm_worker_bonus_per_ant)
        base_growth = config.fungus_farm_base_growth_per_tick + worker_bonus
        compost_available = chamber.inventory.get(ResourceType.COMPOST) > 1e-6
        if compost_available:
            self.last_compost_boost = config.fungus_compost_growth_boost
        requested_growth = base_growth * (1.0 + self.last_compost_boost)

        max_by_leaves = chamber.inventory.get(ResourceType.LEAVES) / max(1e-9, config.fungus_leaf_per_fungus)
        max_by_water = chamber.inventory.get(ResourceType.WATER) / max(1e-9, config.fungus_water_per_fungus)
        max_by_output = chamber.available_capacity(ResourceType.FUNGUS) + max(0.0, 1.0 - self.biomass)
        possible_growth = min(requested_growth, max_by_leaves, max_by_water, max_by_output)
        if possible_growth <= 1e-9:
            self.blocked_reason = "missing_inputs"
            self.last_efficiency = 0.0
            return 0.0

        chamber.remove_resource(ResourceType.LEAVES, possible_growth * config.fungus_leaf_per_fungus)
        chamber.remove_resource(ResourceType.WATER, possible_growth * config.fungus_water_per_fungus)
        if compost_available:
            chamber.remove_resource(ResourceType.COMPOST, possible_growth * config.fungus_compost_per_fungus)

        self.biomass += possible_growth
        self.last_efficiency = possible_growth
        harvested = 0.0
        while self.biomass >= 1.0 and chamber.available_capacity(ResourceType.FUNGUS) > 1e-6:
            added = chamber.add_resource(ResourceType.FUNGUS, 1.0)
            if added <= 1e-6:
                break
            self.biomass -= added
            harvested += added

        if harvested > 0.0:
            self.fungus_harvested += harvested
            self.last_harvest_tick = tick

        if self.output_room_missing(chamber):
            self.blocked_reason = "output_full"
        elif self.input_missing(chamber):
            self.blocked_reason = "missing_inputs"
        else:
            self.blocked_reason = None
        return harvested
