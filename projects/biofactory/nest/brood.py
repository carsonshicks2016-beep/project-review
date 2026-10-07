from __future__ import annotations

from dataclasses import dataclass

from biofactory.resources.resource_types import ResourceType


@dataclass
class NurseryState:
    eggs: int = 0
    larvae: int = 0
    pupae: int = 0
    egg_progress: float = 0.0
    larva_progress: float = 0.0
    pupa_progress: float = 0.0
    workers_born: int = 0
    brood_lost: float = 0.0
    brood_deaths: int = 0
    starvation_ticks: int = 0
    blocked_reason: str | None = "idle"
    last_worker_count: int = 0
    last_efficiency: float = 0.0
    last_birth_tick: int = -999999
    first_larva_hatched: bool = False
    first_pupa_formed: bool = False

    @property
    def total_brood(self) -> int:
        return self.eggs + self.larvae + self.pupae

    def has_room(self, config) -> bool:
        return self.total_brood < config.nursery_capacity

    def input_missing(self, chamber) -> dict[ResourceType, float]:
        missing: dict[ResourceType, float] = {}
        if self.larvae <= 0 and self.pupae <= 0:
            return missing
        if chamber.inventory.get(ResourceType.FOOD) <= 1e-6:
            missing[ResourceType.FOOD] = 1.0
        if chamber.inventory.get(ResourceType.WATER) <= 1e-6:
            missing[ResourceType.WATER] = 1.0
        return missing

    def advance(self, chamber, worker_count: int, config, tick: int = 0) -> int:
        self.last_worker_count = worker_count
        worker_bonus = min(config.nursery_max_worker_bonus, worker_count * config.nursery_worker_bonus_per_ant)
        efficiency = config.nursery_base_efficiency + worker_bonus
        births = 0

        needs_food = self.larvae + self.pupae * config.nursery_pupa_consumption_multiplier
        food_needed = needs_food * config.nursery_food_per_larva_tick
        water_needed = needs_food * config.nursery_water_per_larva_tick
        food_ratio = 1.0
        water_ratio = 1.0
        if food_needed > 0.0:
            consumed_food = chamber.remove_resource(ResourceType.FOOD, food_needed)
            food_ratio = consumed_food / food_needed
        if water_needed > 0.0:
            consumed_water = chamber.remove_resource(ResourceType.WATER, water_needed)
            water_ratio = consumed_water / water_needed

        supply_ratio = min(food_ratio, water_ratio)
        starving = supply_ratio < 0.72 and (self.larvae > 0 or self.pupae > 0)
        if starving:
            self.starvation_ticks += 1
            efficiency *= max(0.0, supply_ratio * 0.55)
        else:
            self.starvation_ticks = 0

        if self.starvation_ticks > config.nursery_starvation_grace_ticks and self.larvae > 0:
            loss = min(float(self.larvae), config.nursery_starvation_loss_per_tick)
            self.brood_lost += loss
            if self.brood_lost >= 1.0:
                lost = min(self.larvae, int(self.brood_lost))
                self.larvae -= lost
                self.brood_lost -= lost
                self.brood_deaths += lost
                chamber.add_resource(ResourceType.WASTE, lost * 0.10)

        self.last_efficiency = efficiency
        if self.eggs > 0 and efficiency > 0.0:
            self.egg_progress += self.eggs * efficiency
            hatched = min(self.eggs, int(self.egg_progress // config.egg_to_larva_ticks))
            if hatched > 0:
                self.eggs -= hatched
                self.larvae += hatched
                self.egg_progress -= hatched * config.egg_to_larva_ticks
                self.first_larva_hatched = True

        if self.larvae > 0 and efficiency > 0.0 and not starving:
            self.larva_progress += self.larvae * efficiency
            formed = min(self.larvae, int(self.larva_progress // config.larva_to_pupa_ticks))
            if formed > 0:
                self.larvae -= formed
                self.pupae += formed
                self.larva_progress -= formed * config.larva_to_pupa_ticks
                self.first_pupa_formed = True

        if self.pupae > 0 and efficiency > 0.0 and not starving:
            self.pupa_progress += self.pupae * efficiency
            births = min(self.pupae, int(self.pupa_progress // config.pupa_to_worker_ticks))
            if births > 0:
                self.pupae -= births
                self.pupa_progress -= births * config.pupa_to_worker_ticks
                self.workers_born += births
                self.last_birth_tick = tick

        if starving:
            if food_ratio <= water_ratio:
                self.blocked_reason = "missing_food"
            else:
                self.blocked_reason = "missing_water"
        elif self.total_brood <= 0:
            self.blocked_reason = "idle"
        else:
            self.blocked_reason = None
        return births
