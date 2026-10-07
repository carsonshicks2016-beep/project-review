from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from biofactory.config import ColonyConfig
from biofactory.colonies.colony_stats import ColonyStats
from biofactory.colonies.genetics import Genes
from biofactory.colonies.territory import TerritoryMap
from biofactory.logistics.demand import ResourceDemandState, calculate_resource_demand
from biofactory.nest.brood import NurseryState
from biofactory.nest.chambers import Chamber
from biofactory.nest.farming import FungusFarmState
from biofactory.nest.processing import ProcessingState
from biofactory.pheromones.pheromone_config import PheromoneType
from biofactory.pheromones.pheromone_layers import PheromoneLayers
from biofactory.resources.inventory import Inventory
from biofactory.resources.logistics import DEMAND_TRACKED_RESOURCES, GATHERABLE_RESOURCES
from biofactory.resources.production_chains import (
    COMPOST_PROCESSING,
    LEAF_FALLBACK_PROCESSING,
    NUTRIENT_PROCESSING,
    PROTEIN_PROCESSING,
)
from biofactory.resources.resource_types import ResourceType

if TYPE_CHECKING:
    from biofactory.agents.ant import Ant
    from biofactory.simulation.events import EventLog


@dataclass
class Colony:
    colony_id: int
    color: tuple[int, int, int]
    nest_x: float
    nest_y: float
    storage_radius: float
    config: ColonyConfig
    inventory: Inventory = field(default_factory=Inventory)
    overflow_inventory: Inventory = field(default_factory=Inventory)
    chambers: list[Chamber] = field(default_factory=list)
    stats: ColonyStats = field(default_factory=ColonyStats)
    territory: TerritoryMap | None = None
    queen_health: float = 1.0
    eggs: int = 0
    larvae: int = 0
    pupae: int = 0
    alive: bool = True
    resource_demand_states: dict[ResourceType, ResourceDemandState] = field(default_factory=dict)
    demand_alert_state: dict[ResourceType, tuple[bool, bool]] = field(default_factory=dict)
    processor_alert_state: dict[int, str | None] = field(default_factory=dict)
    processor_online_ids: set[int] = field(default_factory=set)
    farm_alert_state: dict[int, str | None] = field(default_factory=dict)
    farm_online_ids: set[int] = field(default_factory=set)
    nursery_alert_state: dict[int, str | None] = field(default_factory=dict)
    nursery_online_ids: set[int] = field(default_factory=set)
    first_fungus_harvested: bool = False
    first_egg_laid: bool = False
    first_larva_hatched: bool = False
    first_pupa_formed: bool = False
    first_worker_born: bool = False
    first_batch_outputs: set[ResourceType] = field(default_factory=set)
    queen_egg_progress: float = 0.0
    population_cap_alerted: bool = False
    overall_demand: float = 1.0

    def is_storage_cell(self, x: float, y: float) -> bool:
        return self.chamber_at(x, y) is not None or self.is_overflow_cell(x, y)

    def is_overflow_cell(self, x: float, y: float) -> bool:
        return (x - self.nest_x) ** 2 + (y - self.nest_y) ** 2 <= self.storage_radius * self.storage_radius

    def demand_ratio(self) -> float:
        return self.overall_demand

    def desired_storage_for(self, resource_type: ResourceType) -> float:
        chamber_desired = sum(chamber.desired_for(resource_type) for chamber in self.storage_chambers_for(resource_type))
        if chamber_desired > 0.0:
            return chamber_desired
        return self._configured_desired_storage_for(resource_type)

    def resource_demand(self, resource_type: ResourceType) -> float:
        return self.resource_priority(resource_type)

    def resource_demand_state(self, resource_type: ResourceType) -> ResourceDemandState:
        cached = self.resource_demand_states.get(resource_type)
        if cached is not None:
            return cached
        food_shortage = self._compute_shortage_ratio(ResourceType.FOOD)
        return self._compute_resource_demand_state(resource_type, food_shortage)

    def resource_priority(self, resource_type: ResourceType) -> float:
        return self.resource_demand_state(resource_type).priority

    def resource_saturation(self, resource_type: ResourceType) -> float:
        return self.resource_demand_state(resource_type).saturation

    def critical_resources(self) -> list[ResourceType]:
        states = [self.resource_demand_state(resource_type) for resource_type in DEMAND_TRACKED_RESOURCES]
        return [state.resource_type for state in sorted(states, key=lambda state: state.priority, reverse=True) if state.is_critical]

    def biggest_shortage(self) -> ResourceDemandState | None:
        states = [self.resource_demand_state(resource_type) for resource_type in DEMAND_TRACKED_RESOURCES]
        if not states:
            return None
        return max(states, key=lambda state: state.priority)

    def storage_chambers_for(self, resource_type: ResourceType) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.is_storage and chamber.accepts(resource_type)]

    def accepting_chambers_for(self, resource_type: ResourceType) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.accepts(resource_type)]

    def processor_chambers(self) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.is_processor]

    def fungus_farm_chambers(self) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.is_fungus_farm]

    def production_chambers(self) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.is_production]

    def nursery_chambers(self) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.is_nursery]

    def input_consumer_chambers(self) -> list[Chamber]:
        return [chamber for chamber in self.chambers if chamber.is_input_consumer]

    def processor_summary(self) -> tuple[int, int, int]:
        active = 0
        blocked = 0
        batches = 0
        for chamber in self.processor_chambers():
            state = chamber.processing_state
            if state is None:
                continue
            batches += state.batches_completed
            if state.blocked_reason in ("missing_inputs", "output_full", "damaged", "no_recipe"):
                blocked += 1
            else:
                active += 1
        return active, blocked, batches

    def farm_summary(self) -> tuple[int, int, float]:
        active = 0
        blocked = 0
        harvested = 0.0
        for chamber in self.fungus_farm_chambers():
            state = chamber.fungus_farm_state
            if state is None:
                continue
            harvested += state.fungus_harvested
            if state.blocked_reason in ("missing_inputs", "output_full", "damaged"):
                blocked += 1
            else:
                active += 1
        return active, blocked, harvested

    def nursery_summary(self) -> tuple[int, int, int, int]:
        active = 0
        blocked = 0
        workers_born = 0
        brood_deaths = 0
        for chamber in self.nursery_chambers():
            state = chamber.brood_state
            if state is None:
                continue
            workers_born += state.workers_born
            brood_deaths += state.brood_deaths
            if state.blocked_reason in ("missing_food", "missing_water", "starving", "damaged"):
                blocked += 1
            else:
                active += 1
        return active, blocked, workers_born, brood_deaths

    def total_brood(self) -> int:
        return self.eggs + self.larvae + self.pupae

    def chamber_by_id(self, chamber_id: int | None) -> Chamber | None:
        if chamber_id is None:
            return None
        for chamber in self.chambers:
            if chamber.chamber_id == chamber_id:
                return chamber
        return None

    def chamber_at(self, x: float, y: float) -> Chamber | None:
        best: Chamber | None = None
        best_dist = float("inf")
        for chamber in self.chambers:
            if not chamber.contains(x, y):
                continue
            dist = (x - chamber.x) ** 2 + (y - chamber.y) ** 2
            if dist < best_dist:
                best = chamber
                best_dist = dist
        return best

    def storage_fullness(self, resource_type: ResourceType) -> float:
        capacity = self.storage_capacity(resource_type)
        if capacity <= 0.0:
            return 1.0
        return max(0.0, min(1.0, self.stored_amount(resource_type) / capacity))

    def stored_amount(self, resource_type: ResourceType) -> float:
        return sum(chamber.inventory.get(resource_type) for chamber in self.chambers) + self.overflow_inventory.get(resource_type)

    def storage_capacity(self, resource_type: ResourceType) -> float:
        return sum(chamber.capacity_for(resource_type) for chamber in self.storage_chambers_for(resource_type))

    def available_storage_capacity(self, resource_type: ResourceType) -> float:
        return sum(chamber.available_capacity(resource_type) for chamber in self.storage_chambers_for(resource_type))

    def available_drop_capacity(self, resource_type: ResourceType) -> float:
        return sum(chamber.available_capacity(resource_type) for chamber in self.accepting_chambers_for(resource_type))

    def nearest_chamber_accepting(self, resource_type: ResourceType, x: float, y: float) -> Chamber | None:
        chambers = self.accepting_chambers_for(resource_type)
        if not chambers:
            return None
        return min(chambers, key=lambda chamber: (chamber.x - x) ** 2 + (chamber.y - y) ** 2)

    def best_drop_chamber(self, resource_type: ResourceType, x: float, y: float) -> Chamber | None:
        best: Chamber | None = None
        best_score = -1.0
        for chamber in self.accepting_chambers_for(resource_type):
            if not chamber.can_accept_input(resource_type):
                continue
            distance = float(np.hypot(chamber.x - x, chamber.y - y))
            capacity = max(1.0, chamber.capacity_for(resource_type))
            available_ratio = chamber.available_capacity(resource_type) / capacity
            if chamber.is_input_consumer:
                input_demand = chamber.input_demand(resource_type)
                if input_demand <= 1e-6:
                    continue
                desired = max(1.0, chamber.desired_for(resource_type))
                demand_ratio = input_demand / desired
                if chamber.is_production:
                    output_demand = 0.0
                    if chamber.primary_resource is not None:
                        output_demand = self.processor_output_demand(chamber.primary_resource)
                    if output_demand <= 0.02:
                        continue
                    score = 4.6 * demand_ratio + 1.6 * output_demand + 0.7 * available_ratio
                    if chamber.is_fungus_farm and resource_type in (ResourceType.LEAVES, ResourceType.WATER):
                        score += 0.90 * self.resource_priority(ResourceType.FOOD)
                    if chamber.chamber_type == "leaf_fallback_processor":
                        score *= 0.25
                else:
                    score = 4.2 * demand_ratio + 1.1 * available_ratio + 0.60 * self.resource_priority(resource_type)
            else:
                desired = max(1.0, chamber.desired_for(resource_type))
                demand_ratio = chamber.demand(resource_type) / desired
                score = 2.2 * demand_ratio + 1.4 * available_ratio + 0.55 * self.resource_priority(resource_type)
            score -= distance / max(10.0, self.config.queen_signal_radius)
            if score > best_score:
                best = chamber
                best_score = score
        return best

    def processor_input_demand(self, resource_type: ResourceType) -> float:
        return self.internal_input_demand(resource_type)

    def internal_input_demand(self, resource_type: ResourceType) -> float:
        demand = 0.0
        for chamber in self.input_consumer_chambers():
            if not chamber.can_accept_input(resource_type):
                continue
            if chamber.is_production:
                output_type = chamber.primary_resource
                if output_type is not None and self.processor_output_demand(output_type) <= 0.02:
                    continue
            desired = max(1.0, chamber.desired_for(resource_type))
            scale = 0.25 if chamber.chamber_type == "leaf_fallback_processor" else 1.0
            if chamber.is_nursery and chamber.brood_state is not None:
                scale = 1.20 if chamber.brood_state.blocked_reason in ("missing_food", "missing_water") else 0.95
            demand = max(demand, min(1.0, scale * chamber.input_demand(resource_type) / desired))
        return demand

    def processor_output_demand(self, resource_type: ResourceType) -> float:
        if self.available_storage_capacity(resource_type) <= 1e-6:
            return 0.0
        state = self.resource_demand_state(resource_type)
        if state.is_saturated:
            return 0.0
        return max(state.priority, state.shortage_ratio)

    def processor_output_pressure(self, resource_type: ResourceType) -> float:
        storage_room = self.available_storage_capacity(resource_type)
        storage_capacity = max(1.0, self.storage_capacity(resource_type))
        storage_room_ratio = min(1.0, storage_room / storage_capacity)
        storage_priority = self.resource_priority(resource_type)
        output_sources = [
            chamber
            for chamber in self.production_chambers()
            if resource_type in chamber.outputs and chamber.inventory.get(resource_type) > 1e-6
        ]
        buffered = sum(chamber.inventory.get(resource_type) for chamber in output_sources)
        buffer_pressure = min(1.0, buffered / 3.0)
        if storage_room <= 1e-6:
            return 0.0
        return max(storage_priority, 0.25 * storage_room_ratio, buffer_pressure)

    def best_processor_for_input(self, resource_type: ResourceType, x: float, y: float) -> Chamber | None:
        best: Chamber | None = None
        best_score = 0.0
        for chamber in self.input_consumer_chambers():
            if not chamber.can_accept_input(resource_type):
                continue
            demand = chamber.input_demand(resource_type)
            if demand <= 1e-6:
                continue
            output_demand = 0.0
            if chamber.is_production:
                output_type = chamber.primary_resource
                output_demand = self.processor_output_demand(output_type) if output_type is not None else 0.0
                if output_demand <= 0.02:
                    continue
            distance = float(np.hypot(chamber.x - x, chamber.y - y))
            desired = max(1.0, chamber.desired_for(resource_type))
            score = 5.0 * (demand / desired) + 2.0 * output_demand - distance / max(8.0, self.config.queen_signal_radius)
            if chamber.is_fungus_farm and resource_type in (ResourceType.LEAVES, ResourceType.WATER):
                score += 0.90 * self.resource_priority(ResourceType.FOOD)
            if chamber.is_nursery:
                score += 1.15
            if chamber.chamber_type == "leaf_fallback_processor":
                score *= 0.25
            if score > best_score:
                best = chamber
                best_score = score
        return best

    def best_output_source(self, resource_type: ResourceType, x: float, y: float) -> Chamber | None:
        if self.available_storage_capacity(resource_type) <= 1e-6:
            return None
        best: Chamber | None = None
        best_score = 0.0
        for chamber in self.production_chambers():
            if not chamber.can_release_output(resource_type):
                continue
            amount = chamber.inventory.get(resource_type)
            capacity = max(1.0, chamber.capacity_for(resource_type))
            distance = float(np.hypot(chamber.x - x, chamber.y - y))
            pressure = amount / capacity
            score = 4.0 * pressure + amount - distance / max(8.0, self.config.queen_signal_radius)
            if score > best_score:
                best = chamber
                best_score = score
        return best

    def best_storage_for_output(self, resource_type: ResourceType, x: float, y: float) -> Chamber | None:
        best: Chamber | None = None
        best_score = -1.0
        for chamber in self.storage_chambers_for(resource_type):
            if not chamber.can_accept_input(resource_type):
                continue
            distance = float(np.hypot(chamber.x - x, chamber.y - y))
            capacity = max(1.0, chamber.capacity_for(resource_type))
            available_ratio = chamber.available_capacity(resource_type) / capacity
            demand_ratio = chamber.demand(resource_type) / max(1.0, chamber.desired_for(resource_type))
            score = 2.0 * available_ratio + 1.5 * demand_ratio - distance / max(8.0, self.config.queen_signal_radius)
            if score > best_score:
                best = chamber
                best_score = score
        return best

    def update(self, ants: list["Ant"], pheromones: PheromoneLayers, tick: int, events: "EventLog | None" = None) -> int:
        ant_count = len(ants)
        self._consume_colony_maintenance(ant_count)
        self._process_chambers(ants, tick, events)
        births = self._update_reproduction(ants, tick, events)
        self.refresh_aggregate_inventory()
        self._refresh_brood_counts()
        self._refresh_demand_cache()
        if events is not None:
            self._record_demand_threshold_events(events, tick)
        demand = self.overall_demand
        emission = self.config.base_demand_emission * (0.15 + demand)
        pheromones.add_at(PheromoneType.DEMAND, self.nest_x, self.nest_y, emission, radius=max(2, int(self.storage_radius)))
        self._emit_chamber_demands(pheromones)
        if tick % 25 == 0 and self.critical_resources():
            pheromones.add_at(PheromoneType.DEMAND, self.nest_x, self.nest_y, emission * 2.45, radius=1)
        return births

    def storage_signal_at(self, x: float, y: float) -> float:
        dist = np.hypot(x - self.nest_x, y - self.nest_y)
        return float(np.exp(-dist / max(1.0, self.config.queen_signal_radius)))

    def add_to_storage(self, resource_type: ResourceType, amount: float, tick: int, x: float | None = None, y: float | None = None) -> float:
        if amount <= 0.0:
            return 0.0
        if tick < 0:
            return self._store_storage_only(resource_type, amount)
        sx = self.nest_x if x is None else x
        sy = self.nest_y if y is None else y
        remaining = amount
        local_chamber = self.chamber_at(sx, sy)
        if local_chamber is not None and local_chamber.can_accept_input(resource_type):
            remaining -= local_chamber.add_resource(resource_type, remaining)
        if remaining > 1e-6 and self.available_storage_capacity(resource_type) <= 1e-6 and self.is_overflow_cell(sx, sy):
            self.overflow_inventory.add(resource_type, remaining)
            accepted = amount
            self.refresh_aggregate_inventory()
            if tick >= 0:
                self.stats.record_delivery(tick, resource_type, accepted)
            return accepted
        chamber = self.best_drop_chamber(resource_type, sx, sy)
        if remaining > 1e-6 and chamber is not None and chamber is not local_chamber:
            remaining -= chamber.add_resource(resource_type, remaining)
        if remaining > 1e-6:
            for fallback in sorted(
                self.accepting_chambers_for(resource_type),
                key=lambda candidate: (candidate.x - sx) ** 2 + (candidate.y - sy) ** 2,
            ):
                if remaining <= 1e-6:
                    break
                if fallback.can_accept_input(resource_type):
                    remaining -= fallback.add_resource(resource_type, remaining)
        accepted = amount - max(0.0, remaining)
        if remaining > 1e-6:
            self.overflow_inventory.add(resource_type, remaining)
            accepted += remaining
        self.refresh_aggregate_inventory()
        if accepted > 0.0 and tick >= 0:
            self.stats.record_delivery(tick, resource_type, accepted)
        return accepted

    def _store_storage_only(self, resource_type: ResourceType, amount: float) -> float:
        remaining = amount
        for chamber in sorted(self.storage_chambers_for(resource_type), key=lambda candidate: candidate.fullness(resource_type)):
            if remaining <= 1e-6:
                break
            remaining -= chamber.add_resource(resource_type, remaining)
        accepted = amount - max(0.0, remaining)
        if remaining > 1e-6:
            self.overflow_inventory.add(resource_type, remaining)
            accepted += remaining
        self.refresh_aggregate_inventory()
        return accepted

    def _consume_colony_maintenance(self, ant_count: int) -> None:
        consumed = self.remove_from_storage(ResourceType.FOOD, ant_count * self.config.food_consumption_per_ant)
        self.stats.food_consumed += consumed
        water = self.remove_from_storage(ResourceType.WATER, ant_count * self.config.water_consumption_per_ant)
        protein = self.remove_from_storage(ResourceType.PROTEIN, ant_count * self.config.protein_consumption_per_ant)
        waste = ant_count * self.config.waste_generation_per_ant
        self.store_internal(ResourceType.WASTE, waste)
        self.stats.water_consumed += water
        self.stats.protein_consumed += protein
        self.stats.waste_generated += waste
        if consumed <= ant_count * self.config.food_consumption_per_ant * 0.25:
            self.queen_health = max(0.0, self.queen_health - 0.0003)
        else:
            self.queen_health = min(1.0, self.queen_health + 0.0002)

    def _process_chambers(self, ants: list["Ant"], tick: int, events: "EventLog | None") -> None:
        for chamber in self.fungus_farm_chambers():
            state = chamber.fungus_farm_state
            if state is None:
                continue
            if events is not None and chamber.chamber_id not in self.farm_online_ids:
                self.farm_online_ids.add(chamber.chamber_id)
                events.add(tick, "Fungus farm online")

            harvested = chamber.farm_tick(self._nearby_worker_count(chamber, ants, self.config.fungus_farm_worker_radius), self.config, tick)
            if harvested > 0.0 and events is not None and not self.first_fungus_harvested:
                self.first_fungus_harvested = True
                events.add(tick, "First fungus harvested")
            if events is not None:
                self._record_farm_state_event(chamber, tick, events)

        for chamber in self.processor_chambers():
            state = chamber.processing_state
            if state is None:
                continue
            if events is not None and chamber.chamber_id not in self.processor_online_ids:
                self.processor_online_ids.add(chamber.chamber_id)
                events.add(tick, f"{chamber.chamber_type.replace('_', ' ')} online")

            before_batches = state.batches_completed
            worker_count = self._nearby_worker_count(chamber, ants, self.config.processor_worker_radius)
            completed = chamber.process_tick(worker_count, self.config)
            if completed and state.batches_completed > before_batches:
                state.last_completed_tick = tick
                for output_type, produced in state.recipe.outputs.items():
                    if output_type == ResourceType.FOOD:
                        self.stats.food_produced += produced
                    if events is not None and output_type not in self.first_batch_outputs:
                        self.first_batch_outputs.add(output_type)
                        label = {
                            ResourceType.FOOD: "food",
                            ResourceType.PROTEIN_PASTE: "protein paste",
                            ResourceType.COMPOST: "compost",
                            ResourceType.FUNGUS: "fungus",
                        }.get(output_type, output_type.value)
                        events.add(tick, f"First {label} batch produced")

            if events is not None:
                self._record_processor_state_event(chamber, tick, events)

    def _update_reproduction(self, ants: list["Ant"], tick: int, events: "EventLog | None") -> int:
        for chamber in self.nursery_chambers():
            if events is not None and chamber.chamber_id not in self.nursery_online_ids:
                self.nursery_online_ids.add(chamber.chamber_id)
                events.add(tick, "Nursery online")

        self._try_lay_queen_egg(len(ants), tick, events)
        births = 0
        for chamber in self.nursery_chambers():
            state = chamber.brood_state
            if state is None:
                continue
            worker_count = self._nearby_worker_count(chamber, ants, self.config.nursery_worker_radius)
            for ant in ants:
                if (ant.x - chamber.x) ** 2 + (ant.y - chamber.y) ** 2 <= self.config.nursery_worker_radius**2:
                    if state.total_brood > 0:
                        ant.nursing_ticks += 1
            before_larvae = state.larvae
            had_larva_event = state.first_larva_hatched
            had_pupa_event = state.first_pupa_formed
            born = chamber.nursery_tick(worker_count, self.config, tick)
            if before_larvae > 0 and state.last_efficiency > 0.0:
                for ant in ants:
                    if (ant.x - chamber.x) ** 2 + (ant.y - chamber.y) ** 2 <= self.config.nursery_worker_radius**2:
                        ant.larvae_helped += min(0.02, state.last_efficiency * 0.02)
            births += born
            if events is not None:
                if state.first_larva_hatched and not had_larva_event and not self.first_larva_hatched:
                    self.first_larva_hatched = True
                    events.add(tick, "First larva hatched")
                if state.first_pupa_formed and not had_pupa_event and not self.first_pupa_formed:
                    self.first_pupa_formed = True
                    events.add(tick, "First pupa formed")
                if born > 0 and not self.first_worker_born:
                    self.first_worker_born = True
                    events.add(tick, "First worker born")
                self._record_nursery_state_event(chamber, tick, events)
        return births

    def _try_lay_queen_egg(self, ant_count: int, tick: int, events: "EventLog | None") -> None:
        if self.queen_health < self.config.queen_min_health_to_lay:
            self.queen_egg_progress = 0.0
            return
        if ant_count + self.total_brood() >= self.config.max_population:
            if events is not None and not self.population_cap_alerted:
                self.population_cap_alerted = True
                events.add(tick, "Population cap reached")
            return
        nursery = self._best_nursery_for_egg()
        if nursery is None:
            return
        if self._stored_in_storage(ResourceType.FOOD) < self.config.queen_food_per_egg:
            return
        if self._stored_in_storage(ResourceType.WATER) < self.config.queen_water_per_egg:
            return
        self.population_cap_alerted = False
        self.queen_egg_progress += 1.0
        if self.queen_egg_progress < self.config.queen_egg_interval_ticks:
            return
        state = nursery.brood_state
        if state is None:
            return
        self.remove_from_storage(ResourceType.FOOD, self.config.queen_food_per_egg)
        self.remove_from_storage(ResourceType.WATER, self.config.queen_water_per_egg)
        state.eggs += 1
        self.queen_egg_progress -= self.config.queen_egg_interval_ticks
        if events is not None and not self.first_egg_laid:
            self.first_egg_laid = True
            events.add(tick, "First egg laid")

    def _best_nursery_for_egg(self) -> Chamber | None:
        candidates: list[Chamber] = []
        for chamber in self.nursery_chambers():
            state = chamber.brood_state
            if state is not None and state.has_room(self.config):
                candidates.append(chamber)
        if not candidates:
            return None
        return min(candidates, key=lambda chamber: chamber.brood_state.total_brood if chamber.brood_state is not None else 0)

    def _stored_in_storage(self, resource_type: ResourceType) -> float:
        return sum(chamber.inventory.get(resource_type) for chamber in self.storage_chambers_for(resource_type))

    def _refresh_brood_counts(self) -> None:
        self.eggs = 0
        self.larvae = 0
        self.pupae = 0
        for chamber in self.nursery_chambers():
            state = chamber.brood_state
            if state is None:
                continue
            self.eggs += state.eggs
            self.larvae += state.larvae
            self.pupae += state.pupae

    def _record_nursery_state_event(self, chamber: Chamber, tick: int, events: "EventLog") -> None:
        state = chamber.brood_state
        if state is None:
            return
        current = state.blocked_reason if state.blocked_reason in ("missing_food", "missing_water") else None
        previous = self.nursery_alert_state.get(chamber.chamber_id)
        starving = state.starvation_ticks >= self.config.nursery_starvation_grace_ticks
        if starving:
            current = "starving"
        if current == previous:
            return
        self.nursery_alert_state[chamber.chamber_id] = current
        if current == "missing_food":
            events.add(tick, "Nursery blocked: missing food", severity="warning")
        elif current == "missing_water":
            events.add(tick, "Nursery blocked: missing water", severity="warning")
        elif current == "starving":
            events.add(tick, "Nursery starving", severity="warning")
        elif previous is not None:
            events.add(tick, "Nursery recovered")

    def _nearby_worker_count(self, chamber: Chamber, ants: list["Ant"], radius: float) -> int:
        radius_sq = radius * radius
        count = 0
        for ant in ants:
            if (ant.x - chamber.x) ** 2 + (ant.y - chamber.y) ** 2 <= radius_sq:
                count += 1
        return count

    def _record_farm_state_event(self, chamber: Chamber, tick: int, events: "EventLog") -> None:
        state = chamber.fungus_farm_state
        if state is None:
            return
        current = state.blocked_reason if state.blocked_reason in ("missing_inputs", "output_full") else None
        previous = self.farm_alert_state.get(chamber.chamber_id)
        if current == previous:
            return
        self.farm_alert_state[chamber.chamber_id] = current
        if current == "missing_inputs":
            events.add(tick, "Fungus farm blocked: missing inputs", severity="warning")
        elif current == "output_full":
            events.add(tick, "Fungus farm blocked: output full", severity="warning")
        elif previous is not None:
            events.add(tick, "Fungus farm resumed")

    def _record_processor_state_event(self, chamber: Chamber, tick: int, events: "EventLog") -> None:
        state = chamber.processing_state
        if state is None:
            return
        current = state.blocked_reason if state.blocked_reason in ("missing_inputs", "output_full") else None
        previous = self.processor_alert_state.get(chamber.chamber_id)
        if current == previous:
            return
        self.processor_alert_state[chamber.chamber_id] = current
        label = chamber.chamber_type.replace("_", " ")
        if current == "missing_inputs":
            events.add(tick, f"Processor blocked: missing inputs ({label})", severity="warning")
        elif current == "output_full":
            events.add(tick, f"Processor blocked: output full ({label})", severity="warning")
        elif previous is not None:
            events.add(tick, f"Processor resumed ({label})")

    def _refresh_demand_cache(self) -> None:
        food_shortage = self._compute_shortage_ratio(ResourceType.FOOD)
        self.resource_demand_states = {}
        states: dict[ResourceType, ResourceDemandState] = {}
        for resource_type in DEMAND_TRACKED_RESOURCES:
            states[resource_type] = self._compute_resource_demand_state(resource_type, food_shortage)
        self.resource_demand_states = states
        priorities = [state.priority for state in self.resource_demand_states.values()]
        urgencies = [state.urgency for state in self.resource_demand_states.values()]
        priority_average = sum(min(1.0, priority) for priority in priorities) / max(1, len(priorities))
        urgency_peak = max(urgencies, default=0.0)
        critical_boost = 0.15 if self.critical_resources() else 0.0
        self.overall_demand = min(1.0, 0.58 * priority_average + 0.27 * min(1.0, urgency_peak) + 0.15 * food_shortage + critical_boost)

    def _compute_resource_demand_state(self, resource_type: ResourceType, food_shortage: float) -> ResourceDemandState:
        desired = self.desired_storage_for(resource_type)
        current = self.stored_amount(resource_type)
        state = calculate_resource_demand(resource_type, current, desired, food_shortage_ratio=food_shortage)
        processor_demand = self.processor_input_demand(resource_type)
        if processor_demand > 0.0:
            state = replace(
                state,
                urgency=max(state.urgency, processor_demand),
                priority=max(state.priority, min(1.35, state.priority + processor_demand * 0.60)),
            )
        capacity = self.storage_capacity(resource_type)
        if capacity > 0.0:
            return replace(
                state,
                is_saturated=self.available_storage_capacity(resource_type) <= 1e-6,
                saturation=max(0.0, (current - desired) / max(1.0, desired)),
            )
        return state

    def _compute_shortage_ratio(self, resource_type: ResourceType) -> float:
        desired = self.desired_storage_for(resource_type)
        if desired <= 0.0:
            return 0.0
        current = self.stored_amount(resource_type)
        ratio = (desired - current) / desired
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return float(ratio)

    def _record_demand_threshold_events(self, events: "EventLog", tick: int) -> None:
        for resource_type in DEMAND_TRACKED_RESOURCES:
            state = self.resource_demand_state(resource_type)
            previous = self.demand_alert_state.get(resource_type, (False, False))
            was_critical, was_saturated = previous
            if state.is_critical and not was_critical:
                events.add(tick, f"{resource_type.value} critical", severity="warning")
            elif was_critical and not state.is_critical:
                events.add(tick, f"{resource_type.value} recovered")
            if state.is_saturated and not was_saturated:
                events.add(tick, f"{resource_type.value} saturated")
            elif was_saturated and not state.is_saturated:
                events.add(tick, f"{resource_type.value} saturation cleared")
            self.demand_alert_state[resource_type] = (state.is_critical, state.is_saturated)

    def store_internal(self, resource_type: ResourceType, amount: float) -> float:
        stored = self.add_to_storage(resource_type, amount, tick=-1)
        return stored

    def remove_from_storage(self, resource_type: ResourceType, amount: float) -> float:
        remaining = max(0.0, amount)
        removed = 0.0
        for chamber in sorted(self.storage_chambers_for(resource_type), key=lambda candidate: candidate.fullness(resource_type), reverse=True):
            if remaining <= 1e-9:
                break
            moved = chamber.remove_resource(resource_type, remaining)
            removed += moved
            remaining -= moved
        if remaining > 1e-9:
            moved = self.overflow_inventory.remove(resource_type, remaining)
            removed += moved
            remaining -= moved
        self.refresh_aggregate_inventory()
        return removed

    def refresh_aggregate_inventory(self) -> None:
        amounts: dict[ResourceType, float] = {}
        for chamber in self.chambers:
            for resource_type, amount in chamber.inventory.amounts.items():
                amounts[resource_type] = amounts.get(resource_type, 0.0) + amount
        for resource_type, amount in self.overflow_inventory.amounts.items():
            amounts[resource_type] = amounts.get(resource_type, 0.0) + amount
        self.inventory.amounts = amounts

    def _emit_chamber_demands(self, pheromones: PheromoneLayers) -> None:
        for chamber in self.chambers:
            if chamber.is_input_consumer:
                if chamber.is_nursery:
                    self._emit_nursery_demand(chamber, pheromones)
                    continue
                self._emit_production_demand(chamber, pheromones)
                continue
            resource_type = chamber.primary_resource
            if resource_type is None or not chamber.can_accept_input(resource_type):
                continue
            state = self.resource_demand_state(resource_type)
            if state.priority < 0.05:
                continue
            fullness = chamber.fullness(resource_type)
            amount = self.config.base_demand_emission * state.priority * (0.18 + 0.82 * (1.0 - fullness))
            if state.is_critical:
                amount *= 1.55
            pheromones.add_at(PheromoneType.DEMAND, chamber.x, chamber.y, amount, radius=max(1, int(chamber.radius)))

    def _emit_production_demand(self, chamber: Chamber, pheromones: PheromoneLayers) -> None:
        output_type = chamber.primary_resource
        output_demand = self.processor_output_demand(output_type) if output_type is not None else 0.0
        if output_demand <= 0.02:
            return
        if chamber.available_capacity(output_type) <= 1e-6:
            return
        strongest_input = 0.0
        empty_input_bonus = 0.0
        for resource_type in chamber.accepted_inputs:
            if not chamber.can_accept_input(resource_type):
                continue
            desired = max(1.0, chamber.desired_for(resource_type))
            missing_ratio = chamber.input_demand(resource_type) / desired
            strongest_input = max(strongest_input, missing_ratio)
            if chamber.inventory.get(resource_type) <= 1e-6:
                empty_input_bonus = max(empty_input_bonus, 0.25)
        if strongest_input <= 0.02:
            return
        scale = 0.25 if chamber.chamber_type == "leaf_fallback_processor" else 1.0
        amount = self.config.base_demand_emission * output_demand * (0.35 + strongest_input + empty_input_bonus)
        amount *= scale
        pheromones.add_at(PheromoneType.DEMAND, chamber.x, chamber.y, amount, radius=max(1, int(chamber.radius)))

    def _emit_nursery_demand(self, chamber: Chamber, pheromones: PheromoneLayers) -> None:
        state = chamber.brood_state
        if state is None:
            return
        strongest_input = 0.0
        empty_input_bonus = 0.0
        for resource_type in (ResourceType.FOOD, ResourceType.WATER):
            if not chamber.can_accept_input(resource_type):
                continue
            desired = max(1.0, chamber.desired_for(resource_type))
            missing_ratio = chamber.input_demand(resource_type) / desired
            strongest_input = max(strongest_input, missing_ratio)
            if chamber.inventory.get(resource_type) <= 1e-6:
                empty_input_bonus = max(empty_input_bonus, 0.25)
        if strongest_input <= 0.02:
            return
        brood_pressure = min(1.0, (state.eggs * 0.25 + state.larvae + state.pupae * 0.70) / max(1.0, self.config.nursery_capacity * 0.30))
        amount = self.config.base_demand_emission * (0.35 + brood_pressure) * (0.35 + strongest_input + empty_input_bonus)
        if state.blocked_reason in ("missing_food", "missing_water"):
            amount *= 1.45
        pheromones.add_at(PheromoneType.DEMAND, chamber.x, chamber.y, amount, radius=max(1, int(chamber.radius)))

    def _configured_desired_storage_for(self, resource_type: ResourceType) -> float:
        if resource_type == ResourceType.LEAVES:
            return self.config.desired_leaf_storage
        if resource_type == ResourceType.SEEDS:
            return self.config.desired_seed_storage
        if resource_type == ResourceType.WATER:
            return self.config.desired_water_storage
        if resource_type == ResourceType.PROTEIN:
            return self.config.desired_protein_storage
        if resource_type == ResourceType.WOOD_FIBER:
            return self.config.desired_wood_fiber_storage
        if resource_type == ResourceType.MINERALS:
            return self.config.desired_mineral_storage
        if resource_type == ResourceType.SOIL:
            return self.config.desired_soil_storage
        if resource_type == ResourceType.RESIN:
            return self.config.desired_resin_storage
        if resource_type == ResourceType.DEAD_INSECTS:
            return self.config.desired_dead_insect_storage
        if resource_type == ResourceType.DEAD_ANTS:
            return self.config.desired_dead_ant_storage
        if resource_type == ResourceType.WASTE:
            return self.config.desired_waste_storage
        if resource_type == ResourceType.FOOD:
            return self.config.desired_food_storage
        if resource_type == ResourceType.FUNGUS_SUBSTRATE:
            return self.config.desired_fungus_substrate_storage
        if resource_type == ResourceType.FUNGUS:
            return self.config.desired_fungus_storage
        if resource_type == ResourceType.NUTRIENT_PASTE:
            return self.config.desired_nutrient_paste_storage
        if resource_type == ResourceType.PROTEIN_PASTE:
            return self.config.desired_protein_paste_storage
        if resource_type == ResourceType.LARVAE_FOOD:
            return self.config.desired_larvae_food_storage
        if resource_type == ResourceType.REINFORCED_SOIL:
            return self.config.desired_reinforced_soil_storage
        if resource_type == ResourceType.STRUCTURAL_RESIN:
            return self.config.desired_structural_resin_storage
        if resource_type == ResourceType.COMPOST:
            return self.config.desired_compost_storage
        if resource_type == ResourceType.FERTILIZER:
            return self.config.desired_fertilizer_storage
        if resource_type == ResourceType.SOLDIER_FEED:
            return self.config.desired_soldier_feed_storage
        return 0.0

    @classmethod
    def create_mvp(cls, width: int, height: int, config: ColonyConfig, world_config) -> "Colony":
        initial_inventory = Inventory(
            {
                ResourceType.LEAVES: 120.0,
                ResourceType.FOOD: 80.0,
                ResourceType.WATER: 45.0,
                ResourceType.PROTEIN: 18.0,
                ResourceType.WASTE: 0.0,
            }
        )
        colony = cls(
            colony_id=1,
            color=(74, 210, 255),
            nest_x=world_config.nest_x,
            nest_y=world_config.nest_y,
            storage_radius=world_config.storage_radius,
            config=config,
            inventory=Inventory(),
            chambers=create_mvp_storage_chambers(width, height, config, world_config),
            territory=TerritoryMap.create(width, height),
        )
        for resource_type, amount in initial_inventory.amounts.items():
            colony.add_to_storage(resource_type, amount, tick=-1)
        colony.refresh_aggregate_inventory()
        colony._refresh_demand_cache()
        return colony

    def genetic_template(self, rng) -> Genes:
        return Genes.random(rng)


def create_mvp_storage_chambers(width: int, height: int, config: ColonyConfig, world_config) -> list[Chamber]:
    storage_specs = (
        ("leaf_storage", ResourceType.LEAVES, -6.2, 3.6, 0.72, 1.35),
        ("seed_storage", ResourceType.SEEDS, -9.4, -3.3, 0.34, 1.35),
        ("food_storage", ResourceType.FOOD, -1.8, -5.6, 0.58, 1.45),
        ("water_reservoir", ResourceType.WATER, 6.5, -3.4, 0.62, 1.40),
        ("protein_storage", ResourceType.PROTEIN, 5.2, 5.1, 0.55, 1.42),
        ("wood_fiber_storage", ResourceType.WOOD_FIBER, -10.2, 2.5, 0.34, 1.35),
        ("mineral_storage", ResourceType.MINERALS, 10.1, -0.9, 0.34, 1.35),
        ("soil_storage", ResourceType.SOIL, 9.7, -6.5, 0.36, 1.35),
        ("resin_storage", ResourceType.RESIN, -10.4, -6.5, 0.32, 1.35),
        ("dead_insect_storage", ResourceType.DEAD_INSECTS, 8.7, 4.7, 0.33, 1.35),
        ("dead_ant_recovery", ResourceType.DEAD_ANTS, 2.6, 9.9, 0.30, 1.35),
        ("waste_chamber", ResourceType.WASTE, -1.4, 8.6, 0.52, 1.70),
        ("fungus_storage", ResourceType.FUNGUS, -4.8, -6.4, 0.46, 1.50),
        ("protein_paste_storage", ResourceType.PROTEIN_PASTE, 8.4, 8.3, 0.46, 1.50),
        ("compost_storage", ResourceType.COMPOST, -6.0, 9.7, 0.46, 1.50),
    )
    chambers: list[Chamber] = []
    for chamber_id, (chamber_type, resource_type, dx, dy, radius_scale, capacity_scale) in enumerate(storage_specs, start=1):
        desired = _configured_desired_for(config, resource_type)
        x = _clamp(world_config.nest_x + dx, 2.0, width - 2.0)
        y = _clamp(world_config.nest_y + dy, 2.0, height - 2.0)
        radius = max(2.2, world_config.storage_radius * radius_scale)
        chambers.append(
            Chamber(
                chamber_id=chamber_id,
                chamber_type=chamber_type,
                x=x,
                y=y,
                radius=radius,
                accepted_inputs={resource_type},
                desired_inventory_by_resource={resource_type: desired},
                capacity_by_resource={resource_type: desired * capacity_scale},
            )
        )
    next_id = len(chambers) + 1
    farm_x = _clamp(world_config.nest_x - 7.2, 2.0, width - 2.0)
    farm_y = _clamp(world_config.nest_y - 3.0, 2.0, height - 2.0)
    farm_radius = max(2.3, world_config.storage_radius * 0.56)
    chambers.append(
        Chamber(
            chamber_id=next_id,
            chamber_type="fungus_farm",
            x=farm_x,
            y=farm_y,
            radius=farm_radius,
            accepted_inputs={ResourceType.LEAVES, ResourceType.WATER, ResourceType.COMPOST},
            outputs={ResourceType.FUNGUS},
            desired_inventory_by_resource={
                ResourceType.LEAVES: config.fungus_farm_leaf_buffer,
                ResourceType.WATER: config.fungus_farm_water_buffer,
                ResourceType.COMPOST: config.fungus_farm_compost_buffer,
            },
            capacity_by_resource={
                ResourceType.LEAVES: config.fungus_farm_leaf_buffer * 1.35,
                ResourceType.WATER: config.fungus_farm_water_buffer * 1.35,
                ResourceType.COMPOST: config.fungus_farm_compost_buffer * 1.35,
                ResourceType.FUNGUS: config.fungus_farm_output_capacity,
            },
            fungus_farm_state=FungusFarmState(),
        )
    )
    next_id = len(chambers) + 1
    nursery_x = _clamp(world_config.nest_x + 2.8, 2.0, width - 2.0)
    nursery_y = _clamp(world_config.nest_y - 6.2, 2.0, height - 2.0)
    nursery_radius = max(2.4, world_config.storage_radius * 0.52)
    chambers.append(
        Chamber(
            chamber_id=next_id,
            chamber_type="nursery",
            x=nursery_x,
            y=nursery_y,
            radius=nursery_radius,
            accepted_inputs={ResourceType.FOOD, ResourceType.WATER},
            desired_inventory_by_resource={
                ResourceType.FOOD: config.nursery_food_buffer,
                ResourceType.WATER: config.nursery_water_buffer,
            },
            capacity_by_resource={
                ResourceType.FOOD: config.nursery_food_buffer * 1.35,
                ResourceType.WATER: config.nursery_water_buffer * 1.35,
                ResourceType.WASTE: max(4.0, config.nursery_capacity * 0.18),
            },
            brood_state=NurseryState(),
        )
    )
    processor_specs = (
        ("nutrient_processor", NUTRIENT_PROCESSING, -2.6, -2.7, 0.48),
        ("leaf_fallback_processor", LEAF_FALLBACK_PROCESSING, -8.9, 1.3, 0.40),
        ("protein_processor", PROTEIN_PROCESSING, 6.4, 1.0, 0.46),
        ("compost_processor", COMPOST_PROCESSING, -4.2, 6.7, 0.45),
    )
    next_id = len(chambers) + 1
    for offset, (chamber_type, recipe, dx, dy, radius_scale) in enumerate(processor_specs):
        x = _clamp(world_config.nest_x + dx, 2.0, width - 2.0)
        y = _clamp(world_config.nest_y + dy, 2.0, height - 2.0)
        radius = max(2.0, world_config.storage_radius * radius_scale)
        desired: dict[ResourceType, float] = {}
        capacity: dict[ResourceType, float] = {}
        for resource_type, amount in recipe.inputs.items():
            desired[resource_type] = amount * 2.0
            capacity[resource_type] = max(amount * config.processor_input_capacity_multiplier, amount + 0.5)
        for resource_type, amount in recipe.outputs.items():
            capacity[resource_type] = max(amount * config.processor_output_capacity_multiplier, amount + 0.5)
        chambers.append(
            Chamber(
                chamber_id=next_id + offset,
                chamber_type=chamber_type,
                x=x,
                y=y,
                radius=radius,
                accepted_inputs=set(recipe.inputs),
                outputs=set(recipe.outputs),
                desired_inventory_by_resource=desired,
                capacity_by_resource=capacity,
                processing_state=ProcessingState(recipe),
            )
        )
    return chambers


def _configured_desired_for(config: ColonyConfig, resource_type: ResourceType) -> float:
    if resource_type == ResourceType.LEAVES:
        return config.desired_leaf_storage
    if resource_type == ResourceType.SEEDS:
        return config.desired_seed_storage
    if resource_type == ResourceType.WATER:
        return config.desired_water_storage
    if resource_type == ResourceType.PROTEIN:
        return config.desired_protein_storage
    if resource_type == ResourceType.WOOD_FIBER:
        return config.desired_wood_fiber_storage
    if resource_type == ResourceType.MINERALS:
        return config.desired_mineral_storage
    if resource_type == ResourceType.SOIL:
        return config.desired_soil_storage
    if resource_type == ResourceType.RESIN:
        return config.desired_resin_storage
    if resource_type == ResourceType.DEAD_INSECTS:
        return config.desired_dead_insect_storage
    if resource_type == ResourceType.DEAD_ANTS:
        return config.desired_dead_ant_storage
    if resource_type == ResourceType.WASTE:
        return config.desired_waste_storage
    if resource_type == ResourceType.FOOD:
        return config.desired_food_storage
    if resource_type == ResourceType.FUNGUS_SUBSTRATE:
        return config.desired_fungus_substrate_storage
    if resource_type == ResourceType.FUNGUS:
        return config.desired_fungus_storage
    if resource_type == ResourceType.NUTRIENT_PASTE:
        return config.desired_nutrient_paste_storage
    if resource_type == ResourceType.PROTEIN_PASTE:
        return config.desired_protein_paste_storage
    if resource_type == ResourceType.LARVAE_FOOD:
        return config.desired_larvae_food_storage
    if resource_type == ResourceType.REINFORCED_SOIL:
        return config.desired_reinforced_soil_storage
    if resource_type == ResourceType.STRUCTURAL_RESIN:
        return config.desired_structural_resin_storage
    if resource_type == ResourceType.COMPOST:
        return config.desired_compost_storage
    if resource_type == ResourceType.FERTILIZER:
        return config.desired_fertilizer_storage
    if resource_type == ResourceType.SOLDIER_FEED:
        return config.desired_soldier_feed_storage
    return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
