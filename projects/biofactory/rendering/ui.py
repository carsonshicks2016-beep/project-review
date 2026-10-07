from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pygame

from biofactory.agents.ant import Ant
from biofactory.nest.chambers import Chamber
from biofactory.pheromones.pheromone_config import PHEROMONE_SPECS, PheromoneType
from biofactory.rendering.colors import PANEL_BG, PANEL_LINE, TEXT, TEXT_DIM, WARNING
from biofactory.resources.resource_types import RESOURCE_SPECS, ResourceType
from biofactory.simulation.engine import SimulationEngine
from biofactory.simulation.terrain import TERRAIN_PROPERTIES, TerrainType


@dataclass
class Selection:
    ant_id: int | None = None
    chamber_id: int | None = None
    cell: tuple[int, int] | None = None


class StatsPanel:
    def __init__(self, width: int) -> None:
        self.width = width
        self.font = pygame.font.SysFont("Menlo", 14)
        self.small_font = pygame.font.SysFont("Menlo", 12)
        self.header_font = pygame.font.SysFont("Menlo", 16, bold=True)

    def draw(
        self,
        screen: pygame.Surface,
        engine: SimulationEngine,
        paused: bool,
        speed: int,
        overlays,
        selection: Selection,
        render_timings: dict[str, float] | None = None,
    ) -> None:
        rect = pygame.Rect(screen.get_width() - self.width, 0, self.width, screen.get_height())
        panel = pygame.Surface((self.width, screen.get_height()), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, 218))
        screen.blit(panel, rect)
        pygame.draw.line(screen, PANEL_LINE, (rect.left, 0), (rect.left, rect.height), 1)

        x = rect.left + 14
        y = 14
        y = self._text(screen, "BioFactory MVP", x, y, self.header_font, TEXT)
        snapshot = engine.snapshot()
        state = "paused" if paused else f"{speed}x"
        demand_text = _resource_demand_text(engine)
        shortage = engine.colony.biggest_shortage()
        shortage_text = "none" if shortage is None else f"{shortage.resource_type.value} {shortage.priority:.2f}"
        critical_text = _critical_resource_text(engine)
        fullness_text = _storage_fullness_text(engine)
        active_processors, blocked_processors, batches_completed = engine.colony.processor_summary()
        active_farms, blocked_farms, fungus_harvested = engine.colony.farm_summary()
        active_nurseries, blocked_nurseries, workers_born, brood_deaths = engine.colony.nursery_summary()
        rows = [
            f"mode {overlays.visual_mode.title()}  tick {snapshot.tick}",
            f"quality {overlays.render_quality.title()}",
            f"state {state}  {engine.seasons.name} / {engine.weather.state}",
            f"population {snapshot.population}/{engine.config.colony.max_population}",
            f"brood E/L/P {engine.colony.eggs:3d} {engine.colony.larvae:3d} {engine.colony.pupae:3d}",
            f"L/W/P/X store {snapshot.leaves_stored:4.0f} {snapshot.water_stored:4.0f} {snapshot.protein_stored:4.0f} {snapshot.waste_stored:4.0f}",
            "Sd/Wd/Mn/So/Rs "
            f"{snapshot.stored_resources.get(ResourceType.SEEDS, 0.0):3.0f} "
            f"{snapshot.stored_resources.get(ResourceType.WOOD_FIBER, 0.0):3.0f} "
            f"{snapshot.stored_resources.get(ResourceType.MINERALS, 0.0):3.0f} "
            f"{snapshot.stored_resources.get(ResourceType.SOIL, 0.0):3.0f} "
            f"{snapshot.stored_resources.get(ResourceType.RESIN, 0.0):3.0f}",
            "Dead I/A    "
            f"{snapshot.stored_resources.get(ResourceType.DEAD_INSECTS, 0.0):4.0f} "
            f"{snapshot.stored_resources.get(ResourceType.DEAD_ANTS, 0.0):4.0f}",
            f"Food/Fung/Paste {snapshot.food_stored:4.0f} {snapshot.stored_resources.get(ResourceType.FUNGUS, 0.0):4.0f} {snapshot.stored_resources.get(ResourceType.PROTEIN_PASTE, 0.0):4.0f}",
            f"Compost      {snapshot.stored_resources.get(ResourceType.COMPOST, 0.0):4.0f}",
            f"L/W/P/X demand {demand_text}",
            f"L/Fg/F/W/P/X full {fullness_text}",
            f"biggest {shortage_text}",
            f"critical {critical_text}",
            f"nursery      {active_nurseries} active {blocked_nurseries} blocked",
            f"born/lost    {workers_born:4d} {brood_deaths:4d}",
            f"farms        {active_farms} active {blocked_farms} blocked",
            f"fungus made  {fungus_harvested:5.1f}",
            f"processors   {active_processors} active {blocked_processors} blocked",
            f"batches      {batches_completed}",
            f"L/W/P/X world {snapshot.world_resources[ResourceType.LEAVES]:4.0f} {snapshot.world_resources[ResourceType.WATER]:4.0f} {snapshot.world_resources[ResourceType.PROTEIN]:4.0f} {snapshot.world_resources[ResourceType.WASTE]:4.0f}",
            "Mat world    "
            f"{snapshot.world_resources.get(ResourceType.SEEDS, 0.0):4.0f} "
            f"{snapshot.world_resources.get(ResourceType.WOOD_FIBER, 0.0):4.0f} "
            f"{snapshot.world_resources.get(ResourceType.MINERALS, 0.0):4.0f} "
            f"{snapshot.world_resources.get(ResourceType.SOIL, 0.0):4.0f}",
            f"deliveries    {snapshot.deliveries}",
            f"delivery rate {snapshot.routing.delivery_rate_per_10s:5.2f}/10s",
            f"avg delivery  {snapshot.routing.average_delivery_ticks:5.1f} ticks",
            f"demand        {snapshot.demand_ratio:5.2f}",
            f"traffic cells {snapshot.traffic.congested_cells}",
            f"max traffic   {snapshot.traffic.max_traffic:5.1f}",
        ]
        y = self._rows(screen, rows, x, y + 8)

        overlay_rows = [
            "Overlays",
            "B visual mode",
            "P quality mode",
            f"F food    {'on' if overlays.pheromones[PheromoneType.FOOD] else 'off'}",
            f"I water   {'on' if overlays.pheromones[PheromoneType.WATER] else 'off'}",
            f"O protein {'on' if overlays.pheromones[PheromoneType.PROTEIN] else 'off'}",
            f"V waste   {'on' if overlays.pheromones[PheromoneType.WASTE] else 'off'}",
            f"D demand  {'on' if overlays.pheromones[PheromoneType.DEMAND] else 'off'}",
            f"T traffic {'on' if overlays.pheromones[PheromoneType.TRAFFIC] else 'off'}",
            f"R sources {'on' if overlays.resources else 'off'}",
            f"H heat    {'on' if overlays.traffic_heatmap else 'off'}",
        ]
        y = self._rows(screen, overlay_rows, x, y + 14)
        y = self._legend(screen, overlays, x, y + 3)

        if render_timings:
            frame_ms = render_timings.get("frame", 0.0) * 1000.0
            fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
            timing_rows = [
                "Render",
                f"frame {frame_ms:5.1f} ms  {fps:4.1f} fps",
                f"sim {render_timings.get('sim', 0.0) * 1000.0:5.1f} ms",
                f"base {render_timings.get('base', 0.0) * 1000.0:5.1f}  pher {render_timings.get('pheromones', 0.0) * 1000.0:5.1f}",
                f"ants {render_timings.get('ants', 0.0) * 1000.0:5.1f}  ui {render_timings.get('ui', 0.0) * 1000.0:5.1f}",
            ]
            y = self._rows(screen, timing_rows, x, y + 9, color=TEXT_DIM)

        y = self._text(screen, "Selection", x, y + 10, self.header_font, TEXT)
        selected_ant = _selected_ant(engine, selection.ant_id)
        selected_chamber = engine.colony.chamber_by_id(selection.chamber_id)
        if selected_ant is not None:
            y = self._rows(screen, _ant_rows(selected_ant), x, y + 6)
        elif selected_chamber is not None:
            y = self._rows(screen, _chamber_rows(engine, selected_chamber), x, y + 6)
        elif selection.cell is not None:
            y = self._rows(screen, _cell_rows(engine, selection.cell), x, y + 6)
        else:
            y = self._rows(screen, ["click an ant, chamber, or cell"], x, y + 6, color=TEXT_DIM)

        y = self._text(screen, "Events", x, y + 12, self.header_font, TEXT)
        for event in engine.events.recent(6):
            color = WARNING if event.severity == "warning" else TEXT_DIM
            y = self._text(screen, f"{event.tick:>6} {event.message}", x, y + 2, self.small_font, color)

        controls = [
            "Controls",
            "Space pause  . step",
            "1 2 3 4 5 speed",
            "B cycle visual mode",
            "P quality  F/I/O/V/D/T overlays",
            "WASD/arrows pan",
            "wheel zoom  C center",
        ]
        self._rows(screen, controls, x, max(y + 14, screen.get_height() - 112), color=TEXT_DIM)

    def _rows(
        self,
        screen: pygame.Surface,
        rows: Iterable[str],
        x: int,
        y: int,
        color: tuple[int, int, int] = TEXT,
    ) -> int:
        for row in rows:
            y = self._text(screen, row, x, y, self.font, color)
        return y

    def _text(
        self,
        screen: pygame.Surface,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
        surface = font.render(text, True, color)
        screen.blit(surface, (x, y))
        return y + surface.get_height() + 3

    def _legend(self, screen: pygame.Surface, overlays, x: int, y: int) -> int:
        items = [
            (PheromoneType.FOOD, "food"),
            (PheromoneType.WATER, "water"),
            (PheromoneType.PROTEIN, "protein"),
            (PheromoneType.WASTE, "waste"),
            (PheromoneType.DEMAND, "demand"),
            (PheromoneType.TRAFFIC, "traffic"),
        ]
        cursor_x = x
        row_y = y
        for pheromone_type, label in items:
            if not overlays.pheromones.get(pheromone_type, False):
                continue
            if cursor_x > x + self.width - 86:
                cursor_x = x
                row_y += 18
            color = PHEROMONE_SPECS[pheromone_type].color
            pygame.draw.circle(screen, color, (cursor_x + 5, row_y + 8), 5)
            text_surface = self.small_font.render(label, True, TEXT_DIM)
            screen.blit(text_surface, (cursor_x + 14, row_y + 2))
            cursor_x += max(64, 12 + text_surface.get_width())
        return row_y + 19


def _selected_ant(engine: SimulationEngine, ant_id: int | None) -> Ant | None:
    if ant_id is None:
        return None
    for ant in engine.ants:
        if ant.ant_id == ant_id:
            return ant
    return None


def _ant_rows(ant: Ant) -> list[str]:
    cargo = "none" if ant.inventory_type is None else f"{ant.inventory_type.value} {ant.inventory_amount:.1f}"
    return [
        f"id {ant.ant_id}",
        f"role {ant.inferred_role}",
        f"age {ant.age:.0f}",
        f"health {ant.health:.2f}  energy {ant.energy:.2f}",
        f"velocity {ant.velocity:.3f}",
        f"cargo {cargo}",
        f"target {ant.target_resource.value}",
        f"target chamber {ant.target_chamber_id if ant.target_chamber_id is not None else 'none'}",
        f"source {ant.cargo_source or 'none'}  dest {ant.cargo_destination_kind or 'none'}",
        f"pickup chamber {ant.pickup_chamber_id if ant.pickup_chamber_id is not None else 'none'}",
        f"deliveries {ant.deliveries}",
        f"delivered {ant.resources_delivered:.1f}",
        f"nursing {ant.nursing_ticks}  larvae {ant.larvae_helped:.2f}",
        f"fungus {ant.fungus_tended:.2f}  allies {ant.allies_helped:.2f}",
        f"dig/build {ant.digging_ticks}/{ant.building_ticks}",
        f"atk/def {ant.attack_intent_ticks}/{ant.defend_ticks}",
        f"distance {ant.distance_traveled:.1f}",
        f"max range {ant.distance_from_nest_max:.1f}",
        f"last delivery {ant.last_delivery_ticks} ticks",
        f"genes speed {ant.genes.speed:.2f}",
        f"genes str/agg {ant.genes.strength:.2f} {ant.genes.aggression:.2f}",
        f"genes fear/cur {ant.genes.fear:.2f} {ant.genes.curiosity:.2f}",
        f"genes scent {ant.genes.pheromone_sensitivity:.2f}",
        f"genes traffic {ant.genes.traffic_tolerance:.2f}",
        f"memory {ant.memory[0]:.2f} {ant.memory[1]:.2f} {ant.memory[2]:.2f} {ant.memory[3]:.2f}",
    ]


def _chamber_rows(engine: SimulationEngine, chamber: Chamber) -> list[str]:
    if chamber.is_nursery:
        return _nursery_rows(chamber)
    if chamber.is_fungus_farm:
        return _fungus_farm_rows(chamber)
    if chamber.is_processor:
        return _processor_rows(chamber)
    resource_type = chamber.primary_resource
    if resource_type is None:
        return [
            f"id {chamber.chamber_id}",
            f"type {chamber.chamber_type}",
            "accepts none",
        ]
    stored = chamber.inventory.get(resource_type)
    desired = chamber.desired_for(resource_type)
    capacity = chamber.capacity_for(resource_type)
    demand = chamber.demand(resource_type)
    priority = engine.colony.resource_priority(resource_type)
    return [
        f"id {chamber.chamber_id}",
        f"type {chamber.chamber_type}",
        f"accepts {RESOURCE_SPECS[resource_type].name}",
        f"stored {stored:.1f}",
        f"desired {desired:.1f}",
        f"capacity {capacity:.1f}",
        f"fullness {chamber.fullness(resource_type):.2f}",
        f"demand {demand:.1f}",
        f"priority {priority:.2f}",
        f"health {chamber.health:.2f}",
    ]


def _processor_rows(chamber: Chamber) -> list[str]:
    state = chamber.processing_state
    if state is None:
        return [
            f"id {chamber.chamber_id}",
            f"type {chamber.chamber_type}",
            "processor missing state",
        ]
    input_rows = [
        f"in {resource_type.value} {chamber.inventory.get(resource_type):.1f}/{chamber.desired_for(resource_type):.1f}"
        for resource_type in state.recipe.inputs
    ]
    output_rows = [
        f"out {resource_type.value} {chamber.inventory.get(resource_type):.1f}/{chamber.capacity_for(resource_type):.1f}"
        for resource_type in state.recipe.outputs
    ]
    rows = [
        f"id {chamber.chamber_id}",
        f"type {chamber.chamber_type}",
        f"recipe {state.recipe.name}",
        *input_rows,
        *output_rows,
        f"progress {state.progress:.1f}/{state.recipe.work:.1f}",
        f"efficiency {state.last_efficiency:.2f}",
        f"nearby workers {state.last_worker_count}",
        f"blocked {state.blocked_reason or 'none'}",
        f"batches {state.batches_completed}",
        f"health {chamber.health:.2f}",
    ]
    return rows


def _nursery_rows(chamber: Chamber) -> list[str]:
    state = chamber.brood_state
    if state is None:
        return [
            f"id {chamber.chamber_id}",
            f"type {chamber.chamber_type}",
            "nursery missing state",
        ]
    rows = [
        f"id {chamber.chamber_id}",
        f"type {chamber.chamber_type}",
        f"food {chamber.inventory.get(ResourceType.FOOD):.1f}/{chamber.desired_for(ResourceType.FOOD):.1f}",
        f"water {chamber.inventory.get(ResourceType.WATER):.1f}/{chamber.desired_for(ResourceType.WATER):.1f}",
        f"eggs {state.eggs}  larvae {state.larvae}",
        f"pupae {state.pupae}  total {state.total_brood}",
        f"egg prog {state.egg_progress:.1f}",
        f"larva prog {state.larva_progress:.1f}",
        f"pupa prog {state.pupa_progress:.1f}",
        f"efficiency {state.last_efficiency:.2f}",
        f"nearby workers {state.last_worker_count}",
        f"blocked {state.blocked_reason or 'none'}",
        f"starve ticks {state.starvation_ticks}",
        f"born {state.workers_born}  lost {state.brood_deaths}",
        f"health {chamber.health:.2f}",
    ]
    return rows


def _fungus_farm_rows(chamber: Chamber) -> list[str]:
    state = chamber.fungus_farm_state
    if state is None:
        return [
            f"id {chamber.chamber_id}",
            f"type {chamber.chamber_type}",
            "farm missing state",
        ]
    input_rows = [
        f"in {resource_type.value} {chamber.inventory.get(resource_type):.1f}/{chamber.desired_for(resource_type):.1f}"
        for resource_type in (ResourceType.LEAVES, ResourceType.WATER, ResourceType.COMPOST)
    ]
    rows = [
        f"id {chamber.chamber_id}",
        f"type {chamber.chamber_type}",
        *input_rows,
        f"out fungus {chamber.inventory.get(ResourceType.FUNGUS):.1f}/{chamber.capacity_for(ResourceType.FUNGUS):.1f}",
        f"biomass {state.biomass:.2f}",
        f"efficiency {state.last_efficiency:.3f}",
        f"compost boost {state.last_compost_boost:.2f}",
        f"nearby workers {state.last_worker_count}",
        f"blocked {state.blocked_reason or 'none'}",
        f"harvested {state.fungus_harvested:.1f}",
        f"health {chamber.health:.2f}",
    ]
    return rows


def _cell_rows(engine: SimulationEngine, cell: tuple[int, int]) -> list[str]:
    cx, cy = cell
    terrain = TerrainType(int(engine.world.terrain[cy, cx]))
    props = TERRAIN_PROPERTIES[terrain]
    resource_labels = (
        ("L", ResourceType.LEAVES),
        ("S", ResourceType.SEEDS),
        ("W", ResourceType.WATER),
        ("P", ResourceType.PROTEIN),
        ("Wd", ResourceType.WOOD_FIBER),
        ("Mn", ResourceType.MINERALS),
        ("So", ResourceType.SOIL),
        ("R", ResourceType.RESIN),
        ("DI", ResourceType.DEAD_INSECTS),
        ("DA", ResourceType.DEAD_ANTS),
        ("X", ResourceType.WASTE),
    )
    rows = [
        f"cell {cx}, {cy}",
        f"terrain {props.name}",
        f"speed {engine.world.movement_speed[cy, cx]:.2f}",
        f"energy cost {engine.world.energy_cost[cy, cx]:.2f}",
        f"visibility {engine.world.visibility[cy, cx]:.2f}",
        f"construct diff {engine.world.construction_difficulty[cy, cx]:.2f}",
        f"spawn chance {engine.world.resource_spawn_chance[cy, cx]:.2f}",
        "resources " + " ".join(f"{label}:{engine.resources.layers[resource_type][cy, cx]:.1f}" for label, resource_type in resource_labels),
        f"food pher {engine.pheromones.layers[PheromoneType.FOOD][cy, cx]:.1f}",
        f"water pher {engine.pheromones.layers[PheromoneType.WATER][cy, cx]:.1f}",
        f"protein pher {engine.pheromones.layers[PheromoneType.PROTEIN][cy, cx]:.1f}",
        f"waste pher {engine.pheromones.layers[PheromoneType.WASTE][cy, cx]:.1f}",
        f"demand {engine.pheromones.layers[PheromoneType.DEMAND][cy, cx]:.1f}",
        f"traffic {engine.pheromones.layers[PheromoneType.TRAFFIC][cy, cx]:.1f}",
        f"temp {engine.world.temperature[cy, cx]:.2f}",
        f"humidity {engine.world.humidity[cy, cx]:.2f}",
    ]
    return rows


def _resource_demand_text(engine: SimulationEngine) -> str:
    resources = (ResourceType.LEAVES, ResourceType.WATER, ResourceType.PROTEIN, ResourceType.WASTE)
    return " ".join(f"{engine.colony.resource_priority(resource_type):.2f}" for resource_type in resources)


def _storage_fullness_text(engine: SimulationEngine) -> str:
    resources = (ResourceType.LEAVES, ResourceType.FUNGUS, ResourceType.FOOD, ResourceType.WATER, ResourceType.PROTEIN, ResourceType.WASTE)
    return " ".join(f"{engine.colony.storage_fullness(resource_type):.2f}" for resource_type in resources)


def _critical_resource_text(engine: SimulationEngine) -> str:
    labels = {
        ResourceType.LEAVES: "L",
        ResourceType.WATER: "W",
        ResourceType.PROTEIN: "P",
        ResourceType.WASTE: "X",
        ResourceType.FOOD: "F",
        ResourceType.FUNGUS: "Fung",
        ResourceType.PROTEIN_PASTE: "Paste",
        ResourceType.COMPOST: "Cmp",
    }
    critical = engine.colony.critical_resources()
    if not critical:
        return "none"
    return ",".join(labels.get(resource_type, resource_type.value) for resource_type in critical)
