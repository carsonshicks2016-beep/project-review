from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pygame

from biofactory.agents.ant import Ant
from biofactory.nest.chambers import Chamber
from biofactory.pheromones.pheromone_config import PHEROMONE_SPECS, PheromoneType
from biofactory.rendering.art_assets import ProceduralArtAssets, surface_from_rgb, surface_from_rgba
from biofactory.rendering.camera import Camera
from biofactory.rendering.colors import (
    ANT_OUTLINE,
    BACKGROUND,
    CARGO_LEAF,
    DEMAND_CORE,
    FOOD_STORE,
    STORAGE,
    blend_color,
)
from biofactory.rendering.lod import AntLODContext, AntRenderLOD
from biofactory.rendering.overlays import OverlayState, RenderQuality, VisualMode
from biofactory.rendering.ui import Selection, StatsPanel
from biofactory.resources.logistics import RAW_RESOURCE_TYPES, RESOURCE_PHEROMONES
from biofactory.resources.resource_types import RESOURCE_SPECS, ResourceType
from biofactory.simulation.engine import SimulationEngine


@dataclass(frozen=True)
class QualityProfile:
    resource_interval: int
    pheromone_interval: int
    smooth_scale: bool
    pulse_enabled: bool
    ant_stride: int
    max_mid_ants: int


@dataclass
class SurfaceCache:
    surface: pygame.Surface | None = None
    tick: int = -1
    key: tuple[object, ...] = field(default_factory=tuple)


QUALITY_PROFILES: dict[RenderQuality, QualityProfile] = {
    "fast": QualityProfile(
        resource_interval=18,
        pheromone_interval=5,
        smooth_scale=False,
        pulse_enabled=False,
        ant_stride=2,
        max_mid_ants=700,
    ),
    "balanced": QualityProfile(
        resource_interval=8,
        pheromone_interval=3,
        smooth_scale=True,
        pulse_enabled=True,
        ant_stride=1,
        max_mid_ants=1500,
    ),
    "quality": QualityProfile(
        resource_interval=3,
        pheromone_interval=1,
        smooth_scale=True,
        pulse_enabled=True,
        ant_stride=1,
        max_mid_ants=3200,
    ),
}


class Renderer:
    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self.overlays = OverlayState(
            visual_mode=engine.config.render.default_visual_mode,
            render_quality=engine.config.render.default_render_quality,
        )
        self.art = ProceduralArtAssets(
            engine.world.terrain,
            seed=engine.config.world.seed,
            art_cell_size=engine.config.render.art_cell_size,
        )
        self.panel = StatsPanel(engine.config.render.ui_width)
        self.last_timings: dict[str, float] = {}
        self.external_timings: dict[str, float] = {}
        self._base_surfaces: dict[VisualMode, pygame.Surface] = {}
        self._base_scaled_cache: dict[tuple[VisualMode, tuple[int, int], RenderQuality], pygame.Surface] = {}
        self._resource_cache = SurfaceCache()
        self._pheromone_cache = SurfaceCache()
        self._grid_phase_x: np.ndarray | None = None
        self._grid_phase_y: np.ndarray | None = None

    def render(
        self,
        screen: pygame.Surface,
        camera: Camera,
        paused: bool,
        speed: int,
        selection: Selection,
    ) -> None:
        frame_start = perf_counter()
        profile = QUALITY_PROFILES[self.overlays.render_quality]
        top_left = camera.world_to_screen(0.0, 0.0)
        scaled_size = self._scaled_world_size(camera)

        screen.fill(BACKGROUND)
        timings = dict(self.external_timings)
        section_start = perf_counter()
        base = self._scaled_base_surface(self.overlays.visual_mode, scaled_size, profile)
        screen.blit(base, top_left)
        timings["base"] = perf_counter() - section_start

        section_start = perf_counter()
        if self.overlays.resources:
            resource_layer = self._resource_surface(scaled_size, profile)
            if resource_layer is not None:
                screen.blit(resource_layer, top_left, special_flags=pygame.BLEND_RGBA_ADD)
        timings["resources"] = perf_counter() - section_start

        section_start = perf_counter()
        pheromone_layer = self._pheromone_surface(scaled_size, profile)
        if pheromone_layer is not None:
            screen.blit(pheromone_layer, top_left, special_flags=pygame.BLEND_ADD)
        timings["pheromones"] = perf_counter() - section_start

        section_start = perf_counter()
        self._draw_storage(screen, camera, selection.chamber_id)
        timings["storage"] = perf_counter() - section_start

        section_start = perf_counter()
        self._draw_ants(screen, camera, selection.ant_id, profile)
        timings["ants"] = perf_counter() - section_start

        section_start = perf_counter()
        self.panel.draw(screen, self.engine, paused, speed, self.overlays, selection, self.last_timings)
        timings["ui"] = perf_counter() - section_start

        timings["frame"] = perf_counter() - frame_start
        self.last_timings = _smooth_timings(self.last_timings, timings)

    def set_external_timing(self, name: str, seconds: float) -> None:
        self.external_timings[name] = seconds

    def _scaled_world_size(self, camera: Camera) -> tuple[int, int]:
        return (
            max(1, int(self.engine.world.width * camera.scale)),
            max(1, int(self.engine.world.height * camera.scale)),
        )

    def _scaled_base_surface(
        self,
        mode: VisualMode,
        scaled_size: tuple[int, int],
        profile: QualityProfile,
    ) -> pygame.Surface:
        key = (mode, scaled_size, self.overlays.render_quality)
        cached = self._base_scaled_cache.get(key)
        if cached is not None:
            return cached

        base = self._base_surface(mode)
        scaled = _scale_surface(base, scaled_size, smooth=profile.smooth_scale)
        self._base_scaled_cache[key] = scaled
        return scaled

    def _base_surface(self, mode: VisualMode) -> pygame.Surface:
        cached = self._base_surfaces.get(mode)
        if cached is not None:
            return cached
        surface = surface_from_rgb(self._base_world_rgb(mode))
        self._base_surfaces[mode] = surface
        return surface

    def _base_world_rgb(self, mode: VisualMode) -> np.ndarray:
        base = self.art.terrain_rgb.astype(np.float32)
        if mode == "natural":
            return base.astype(np.uint8, copy=True)
        if mode == "hybrid":
            dark = np.array((8, 13, 12), dtype=np.float32)
            return np.clip(base * 0.78 + dark * 0.22, 0, 255).astype(np.uint8)

        dark = np.array((4, 8, 9), dtype=np.float32)
        rgb = base * 0.16 + dark * 0.84
        if np.any(self.engine.world.obstacles):
            high_obstacles = self.art.upsample_scalar(self.engine.world.obstacles.astype(np.float32)) > 0.0
            rgb[high_obstacles] = rgb[high_obstacles] * 0.62 + np.array((86, 88, 92), dtype=np.float32) * 0.38
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _resource_surface(self, scaled_size: tuple[int, int], profile: QualityProfile) -> pygame.Surface | None:
        key = (scaled_size, self.overlays.visual_mode, self.overlays.render_quality)
        if self._resource_cache.surface is not None:
            is_fresh = self.engine.tick - self._resource_cache.tick < profile.resource_interval
            if is_fresh and self._resource_cache.key == key:
                return self._resource_cache.surface

        alpha_scale = self._resource_alpha_scale()
        height, width = self.engine.world.height, self.engine.world.width
        rgb_accum = np.zeros((height, width, 3), dtype=np.float32)
        alpha_max = np.zeros((height, width), dtype=np.float32)

        for resource_type in RAW_RESOURCE_TYPES:
            layer = self.engine.resources.layers[resource_type]
            strength = np.sqrt(np.clip(layer / 15.0, 0.0, 1.0))
            scale = alpha_scale * (0.78 if resource_type == ResourceType.WASTE else 1.0)
            alpha = np.clip(strength * scale, 0.0, 1.0)
            if not np.any(alpha):
                continue
            color = np.array(RESOURCE_SPECS[resource_type].color, dtype=np.float32)
            rgb_accum += color * alpha[..., None]
            alpha_max = np.maximum(alpha_max, alpha)

        if not np.any(alpha_max):
            self._resource_cache = SurfaceCache(None, self.engine.tick, key)
            return None

        rgb = np.clip(rgb_accum, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha_max * 255.0, 0, 255).astype(np.uint8)
        surface = surface_from_rgba(rgb, alpha)
        scaled = _scale_surface(surface, scaled_size, smooth=profile.smooth_scale)
        self._resource_cache = SurfaceCache(scaled, self.engine.tick, key)
        return scaled

    def _pheromone_surface(self, scaled_size: tuple[int, int], profile: QualityProfile) -> pygame.Surface | None:
        enabled = tuple(
            pheromone_type
            for pheromone_type in (
                PheromoneType.FOOD,
                PheromoneType.WATER,
                PheromoneType.PROTEIN,
                PheromoneType.WASTE,
                PheromoneType.DEMAND,
                PheromoneType.TRAFFIC,
            )
            if self.overlays.pheromones.get(pheromone_type, False)
        )
        key = (
            scaled_size,
            self.overlays.visual_mode,
            self.overlays.render_quality,
            enabled,
            self.overlays.traffic_heatmap,
        )
        if self._pheromone_cache.surface is not None:
            is_fresh = self.engine.tick - self._pheromone_cache.tick < profile.pheromone_interval
            if is_fresh and self._pheromone_cache.key == key:
                return self._pheromone_cache.surface

        height, width = self.engine.world.height, self.engine.world.width
        rgb_accum = np.zeros((height, width, 3), dtype=np.float32)
        alpha_max = np.zeros((height, width), dtype=np.float32)

        for pheromone_type in enabled:
            layer_rgb, layer_alpha = self._low_res_pheromone_layer(pheromone_type, profile)
            rgb_accum += layer_rgb
            alpha_max = np.maximum(alpha_max, layer_alpha)

        if self.overlays.traffic_heatmap:
            traffic = self.engine.pheromones.layers[PheromoneType.TRAFFIC]
            heat = np.clip(traffic / 40.0, 0.0, 1.0) * 0.34
            color = np.array(PHEROMONE_SPECS[PheromoneType.TRAFFIC].color, dtype=np.float32)
            rgb_accum += color * heat[..., None]
            alpha_max = np.maximum(alpha_max, heat)

        if not np.any(rgb_accum):
            self._pheromone_cache = SurfaceCache(None, self.engine.tick, key)
            return None

        rgb = np.clip(rgb_accum, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha_max * 255.0, 0, 255).astype(np.uint8)
        surface = surface_from_rgba(rgb, alpha)
        scaled = _scale_surface(surface, scaled_size, smooth=profile.smooth_scale)
        self._pheromone_cache = SurfaceCache(scaled, self.engine.tick, key)
        return scaled

    def _low_res_pheromone_layer(
        self,
        pheromone_type: PheromoneType,
        profile: QualityProfile,
    ) -> tuple[np.ndarray, np.ndarray]:
        spec = PHEROMONE_SPECS[pheromone_type]
        layer = self.engine.pheromones.layers[pheromone_type]
        if float(np.max(layer)) <= 0.01:
            empty_rgb = np.zeros((*layer.shape, 3), dtype=np.float32)
            empty_alpha = np.zeros(layer.shape, dtype=np.float32)
            return empty_rgb, empty_alpha

        strength = np.sqrt(np.clip(layer / max(1.0, spec.max_concentration), 0.0, 1.0))
        if profile.pulse_enabled:
            strength = self._apply_trail_pulse(strength, pheromone_type)

        threshold = self._glow_threshold(pheromone_type)
        alpha = np.clip((strength - threshold) / max(0.05, 1.0 - threshold), 0.0, 1.0) ** 0.68
        alpha *= self._glow_alpha_scale(pheromone_type)
        alpha = np.clip(alpha, 0.0, 1.0)
        color = np.array(spec.color, dtype=np.float32)
        rgb = color * alpha[..., None]
        return rgb, alpha

    def _apply_trail_pulse(self, strength: np.ndarray, pheromone_type: PheromoneType) -> np.ndarray:
        if pheromone_type == PheromoneType.TRAFFIC and self.overlays.visual_mode == "natural":
            return strength
        if self._grid_phase_x is None or self._grid_phase_y is None:
            height, width = strength.shape
            yy, xx = np.mgrid[:height, :width]
            self._grid_phase_x = xx.astype(np.float32)
            self._grid_phase_y = yy.astype(np.float32)
        phase = np.sin((self._grid_phase_x * 0.16 + self._grid_phase_y * 0.24) + self.engine.tick * self.engine.config.render.trail_pulse_speed)
        pulse = 0.88 + 0.18 * np.clip(phase, 0.0, 1.0)
        pulsed = strength.copy()
        strong = strength > 0.36
        pulsed[strong] *= pulse[strong]
        return np.clip(pulsed, 0.0, 1.0)

    def _resource_alpha_scale(self) -> float:
        if self.overlays.visual_mode == "natural":
            return 0.46
        if self.overlays.visual_mode == "circuit":
            return 0.10
        return 0.24

    def _pheromone_alpha_scale(self, pheromone_type: PheromoneType) -> float:
        if self.overlays.visual_mode == "natural":
            if pheromone_type == PheromoneType.TRAFFIC:
                return 0.10
            if pheromone_type == PheromoneType.DEMAND:
                return 0.14
            if pheromone_type in _RESOURCE_PHEROMONE_TYPES:
                return 0.18
            return 0.20
        if self.overlays.visual_mode == "circuit":
            if pheromone_type == PheromoneType.TRAFFIC:
                return 0.86
            if pheromone_type == PheromoneType.DEMAND:
                return 0.92
            if pheromone_type in _RESOURCE_PHEROMONE_TYPES:
                return 0.76
            return 0.88
        if pheromone_type == PheromoneType.TRAFFIC:
            return 0.26
        if pheromone_type == PheromoneType.DEMAND:
            return 0.46
        if pheromone_type in _RESOURCE_PHEROMONE_TYPES:
            return 0.50
        return 0.58

    def _glow_threshold(self, pheromone_type: PheromoneType) -> float:
        if self.overlays.visual_mode == "circuit":
            return 0.06 if pheromone_type in _RESOURCE_PHEROMONE_TYPES else 0.10
        if self.overlays.visual_mode == "natural":
            return 0.18 if pheromone_type in _RESOURCE_PHEROMONE_TYPES else 0.24
        return 0.10 if pheromone_type in _RESOURCE_PHEROMONE_TYPES else 0.16

    def _glow_alpha_scale(self, pheromone_type: PheromoneType) -> float:
        mode_scale = self._pheromone_alpha_scale(pheromone_type)
        if self.overlays.render_quality == "fast":
            return mode_scale * 0.78
        if self.overlays.render_quality == "quality":
            return mode_scale * 1.10
        return mode_scale

    def _draw_storage(self, screen: pygame.Surface, camera: Camera, selected_chamber_id: int | None) -> None:
        colony = self.engine.colony
        sx, sy = camera.world_to_screen(colony.nest_x, colony.nest_y)
        radius = max(5, int(colony.storage_radius * camera.scale * 0.40))
        pulse = int(35 + 45 * colony.demand_ratio())
        storage_surface = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
        center = (radius + 3, radius + 3)
        pygame.draw.circle(storage_surface, (*DEMAND_CORE, 30 + pulse), center, radius + max(3, int(camera.scale * 0.5)))
        pygame.draw.circle(storage_surface, (*STORAGE, 55), center, radius)
        pygame.draw.circle(storage_surface, (255, 226, 196, 145), center, max(2, int(radius * 0.35)))
        pygame.draw.circle(storage_surface, (*STORAGE, 230), center, radius, max(1, int(2 * camera.zoom)))
        screen.blit(storage_surface, (sx - radius - 3, sy - radius - 3))
        for chamber in colony.chambers:
            self._draw_chamber(screen, camera, chamber, selected=chamber.chamber_id == selected_chamber_id)
        self._draw_storage_bars(screen, sx, sy, max(6, int(colony.storage_radius * camera.scale)))

    def _draw_chamber(self, screen: pygame.Surface, camera: Camera, chamber: Chamber, selected: bool) -> None:
        if chamber.is_nursery:
            self._draw_nursery_chamber(screen, camera, chamber, selected)
            return
        if chamber.is_fungus_farm:
            self._draw_fungus_farm_chamber(screen, camera, chamber, selected)
            return
        if chamber.is_processor:
            self._draw_processor_chamber(screen, camera, chamber, selected)
            return
        resource_type = chamber.primary_resource
        if resource_type is None:
            return
        sx, sy = camera.world_to_screen(chamber.x, chamber.y)
        radius = max(5, int(chamber.radius * camera.scale))
        color = _resource_color(resource_type)
        fill = chamber.fullness(resource_type)
        state = self.engine.colony.resource_demand_state(resource_type)
        chamber_surface = pygame.Surface((radius * 2 + 12, radius * 2 + 12), pygame.SRCALPHA)
        center = (radius + 6, radius + 6)
        glow_alpha = 30 + int(80 * state.priority)
        if state.is_critical:
            glow_alpha += int(35 + 35 * np.sin(self.engine.tick * 0.12))
        pygame.draw.circle(chamber_surface, (*color, min(190, glow_alpha)), center, radius + 5)
        pygame.draw.circle(chamber_surface, (10, 14, 14, 190), center, radius)
        pygame.draw.circle(chamber_surface, (*color, 55 + int(135 * fill)), center, max(2, int(radius * (0.22 + 0.70 * fill))))
        if state.is_critical:
            pygame.draw.circle(chamber_surface, (*DEMAND_CORE, 190), center, radius + 2, max(1, int(1.5 * camera.zoom)))
        elif state.is_saturated:
            pygame.draw.circle(chamber_surface, (96, 108, 105, 185), center, radius + 1, max(1, int(1.25 * camera.zoom)))
        else:
            pygame.draw.circle(chamber_surface, (*color, 210), center, radius + 1, max(1, int(1.4 * camera.zoom)))
        if selected:
            pygame.draw.circle(chamber_surface, (255, 255, 255, 235), center, radius + 5, max(1, int(1.5 * camera.zoom)))
        screen.blit(chamber_surface, (sx - radius - 6, sy - radius - 6), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_processor_chamber(self, screen: pygame.Surface, camera: Camera, chamber: Chamber, selected: bool) -> None:
        state = chamber.processing_state
        output_type = chamber.primary_resource
        if state is None or output_type is None:
            return
        sx, sy = camera.world_to_screen(chamber.x, chamber.y)
        radius = max(5, int(chamber.radius * camera.scale))
        output_color = _resource_color(output_type)
        progress = float(np.clip(state.progress / max(1.0, state.recipe.work), 0.0, 1.0))
        blocked = state.blocked_reason in ("missing_inputs", "output_full", "damaged", "no_recipe")
        age_since_batch = max(0, self.engine.tick - state.last_completed_tick)
        completion_flash = max(0.0, 1.0 - age_since_batch / 28.0)

        chamber_surface = pygame.Surface((radius * 2 + 16, radius * 2 + 16), pygame.SRCALPHA)
        center = (radius + 8, radius + 8)
        glow_alpha = 35 + int(70 * self.engine.colony.processor_output_pressure(output_type)) + int(90 * completion_flash)
        pygame.draw.circle(chamber_surface, (*output_color, min(220, glow_alpha)), center, radius + 6)
        pygame.draw.circle(chamber_surface, (8, 11, 11, 218), center, radius)
        pygame.draw.circle(chamber_surface, (18, 22, 20, 210), center, max(2, int(radius * 0.72)))

        ring_width = max(1, int(1.6 * camera.zoom))
        pygame.draw.circle(chamber_surface, (*output_color, 220), center, radius + 1, ring_width)
        if progress > 0.0:
            rect = pygame.Rect(center[0] - radius - 2, center[1] - radius - 2, (radius + 2) * 2, (radius + 2) * 2)
            pygame.draw.arc(
                chamber_surface,
                (255, 245, 190, 235),
                rect,
                -np.pi / 2,
                -np.pi / 2 + progress * np.pi * 2,
                max(2, int(2.2 * camera.zoom)),
            )

        inputs = list(state.recipe.inputs)
        pip_radius = max(2, int(radius * 0.16))
        for index, input_type in enumerate(inputs):
            angle = -np.pi / 2 + index * (np.pi * 2 / max(1, len(inputs)))
            pip_x = int(center[0] + np.cos(angle) * radius * 0.78)
            pip_y = int(center[1] + np.sin(angle) * radius * 0.78)
            input_color = _resource_color(input_type)
            fill = float(np.clip(chamber.inventory.get(input_type) / max(1.0, chamber.desired_for(input_type)), 0.0, 1.0))
            pygame.draw.circle(chamber_surface, (4, 7, 7, 220), (pip_x, pip_y), pip_radius + 1)
            pygame.draw.circle(chamber_surface, (*input_color, 75 + int(160 * fill)), (pip_x, pip_y), pip_radius)

        if blocked:
            outline = (164, 82, 58, 220) if state.blocked_reason == "missing_inputs" else (190, 134, 54, 220)
            pygame.draw.circle(chamber_surface, outline, center, radius + 4, max(1, int(1.8 * camera.zoom)))
        if selected:
            pygame.draw.circle(chamber_surface, (255, 255, 255, 235), center, radius + 6, max(1, int(1.5 * camera.zoom)))
        screen.blit(chamber_surface, (sx - radius - 8, sy - radius - 8), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_fungus_farm_chamber(self, screen: pygame.Surface, camera: Camera, chamber: Chamber, selected: bool) -> None:
        state = chamber.fungus_farm_state
        if state is None:
            return
        sx, sy = camera.world_to_screen(chamber.x, chamber.y)
        radius = max(5, int(chamber.radius * camera.scale))
        fungus_color = _resource_color(ResourceType.FUNGUS)
        blocked = state.blocked_reason in ("missing_inputs", "output_full", "damaged")
        harvest_flash = max(0.0, 1.0 - max(0, self.engine.tick - state.last_harvest_tick) / 38.0)
        biomass_fill = float(np.clip(0.18 + 0.70 * state.biomass, 0.18, 0.88))
        output_fill = chamber.fullness(ResourceType.FUNGUS)
        compost_boost = float(np.clip(state.last_compost_boost / max(0.01, self.engine.config.colony.fungus_compost_growth_boost), 0.0, 1.0))

        chamber_surface = pygame.Surface((radius * 2 + 18, radius * 2 + 18), pygame.SRCALPHA)
        center = (radius + 9, radius + 9)
        pulse = 0.5 + 0.5 * np.sin(self.engine.tick * 0.10)
        glow_alpha = 45 + int(55 * output_fill) + int(70 * compost_boost) + int(105 * harvest_flash)
        pygame.draw.circle(chamber_surface, (*fungus_color, min(230, glow_alpha)), center, radius + 7)
        pygame.draw.circle(chamber_surface, (9, 12, 9, 228), center, radius)
        pygame.draw.circle(chamber_surface, (30, 42, 26, 220), center, max(2, int(radius * 0.80)))
        pygame.draw.circle(chamber_surface, (*fungus_color, 70 + int(105 * pulse)), center, max(2, int(radius * biomass_fill)))

        mat_radius = max(2, int(radius * biomass_fill))
        for index in range(7):
            angle = index * (np.pi * 2 / 7.0) + self.engine.tick * 0.006
            wobble = 0.46 + 0.30 * ((index * 37) % 10) / 10.0
            dot_x = int(center[0] + np.cos(angle) * mat_radius * wobble)
            dot_y = int(center[1] + np.sin(angle) * mat_radius * wobble)
            dot_radius = max(1, int(radius * (0.08 + 0.02 * (index % 3))))
            pygame.draw.circle(chamber_surface, (168, 255, 156, 95), (dot_x, dot_y), dot_radius)

        pip_types = (ResourceType.LEAVES, ResourceType.WATER, ResourceType.COMPOST)
        pip_radius = max(2, int(radius * 0.15))
        for index, input_type in enumerate(pip_types):
            angle = -np.pi / 2 + index * (np.pi * 2 / len(pip_types))
            pip_x = int(center[0] + np.cos(angle) * radius * 0.82)
            pip_y = int(center[1] + np.sin(angle) * radius * 0.82)
            input_color = _resource_color(input_type)
            fill = float(np.clip(chamber.inventory.get(input_type) / max(1.0, chamber.desired_for(input_type)), 0.0, 1.0))
            pygame.draw.circle(chamber_surface, (4, 7, 5, 225), (pip_x, pip_y), pip_radius + 1)
            pygame.draw.circle(chamber_surface, (*input_color, 70 + int(165 * fill)), (pip_x, pip_y), pip_radius)

        ring_color = (174, 255, 150, 230)
        if compost_boost > 0.0:
            ring_color = (218, 238, 126, 240)
        pygame.draw.circle(chamber_surface, ring_color, center, radius + 1, max(1, int(1.5 * camera.zoom)))
        if blocked:
            outline = (164, 82, 58, 225) if state.blocked_reason == "missing_inputs" else (190, 134, 54, 225)
            pygame.draw.circle(chamber_surface, outline, center, radius + 4, max(1, int(1.8 * camera.zoom)))
        if selected:
            pygame.draw.circle(chamber_surface, (255, 255, 255, 235), center, radius + 6, max(1, int(1.5 * camera.zoom)))
        screen.blit(chamber_surface, (sx - radius - 9, sy - radius - 9), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_nursery_chamber(self, screen: pygame.Surface, camera: Camera, chamber: Chamber, selected: bool) -> None:
        state = chamber.brood_state
        if state is None:
            return
        sx, sy = camera.world_to_screen(chamber.x, chamber.y)
        radius = max(5, int(chamber.radius * camera.scale))
        food_color = _resource_color(ResourceType.FOOD)
        water_color = _resource_color(ResourceType.WATER)
        brood_color = (255, 196, 130)
        blocked = state.blocked_reason in ("missing_food", "missing_water", "starving", "damaged")
        birth_flash = max(0.0, 1.0 - max(0, self.engine.tick - state.last_birth_tick) / 42.0)
        brood_fill = float(np.clip(state.total_brood / max(1.0, self.engine.config.colony.nursery_capacity), 0.0, 1.0))
        pulse = 0.5 + 0.5 * np.sin(self.engine.tick * 0.11)

        chamber_surface = pygame.Surface((radius * 2 + 18, radius * 2 + 18), pygame.SRCALPHA)
        center = (radius + 9, radius + 9)
        glow_alpha = 42 + int(82 * brood_fill) + int(110 * birth_flash)
        pygame.draw.circle(chamber_surface, (*brood_color, min(235, glow_alpha)), center, radius + 7)
        pygame.draw.circle(chamber_surface, (16, 10, 9, 228), center, radius)
        pygame.draw.circle(chamber_surface, (42, 25, 19, 220), center, max(2, int(radius * 0.78)))
        pygame.draw.circle(chamber_surface, (220, 132, 88, 42 + int(54 * pulse)), center, max(2, int(radius * (0.30 + 0.45 * brood_fill))))

        self._draw_brood_dots(chamber_surface, center, radius, state.eggs, (255, 226, 174, 170), 0.32)
        self._draw_brood_dots(chamber_surface, center, radius, state.larvae, (255, 168, 116, 175), 0.54)
        self._draw_brood_dots(chamber_surface, center, radius, state.pupae, (224, 120, 96, 185), 0.72)

        for index, input_type in enumerate((ResourceType.FOOD, ResourceType.WATER)):
            angle = -np.pi / 2 + index * np.pi
            pip_x = int(center[0] + np.cos(angle) * radius * 0.82)
            pip_y = int(center[1] + np.sin(angle) * radius * 0.82)
            input_color = food_color if input_type == ResourceType.FOOD else water_color
            fill = float(np.clip(chamber.inventory.get(input_type) / max(1.0, chamber.desired_for(input_type)), 0.0, 1.0))
            pip_radius = max(2, int(radius * 0.16))
            pygame.draw.circle(chamber_surface, (6, 4, 4, 225), (pip_x, pip_y), pip_radius + 1)
            pygame.draw.circle(chamber_surface, (*input_color, 75 + int(160 * fill)), (pip_x, pip_y), pip_radius)

        pygame.draw.circle(chamber_surface, (255, 190, 132, 230), center, radius + 1, max(1, int(1.5 * camera.zoom)))
        if blocked:
            outline = (190, 74, 54, 225) if state.starvation_ticks > 0 else (196, 132, 58, 225)
            pygame.draw.circle(chamber_surface, outline, center, radius + 4, max(1, int(1.8 * camera.zoom)))
        if selected:
            pygame.draw.circle(chamber_surface, (255, 255, 255, 235), center, radius + 6, max(1, int(1.5 * camera.zoom)))
        screen.blit(chamber_surface, (sx - radius - 9, sy - radius - 9), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_brood_dots(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        radius: int,
        count: int,
        color: tuple[int, int, int, int],
        ring_scale: float,
    ) -> None:
        visible_count = min(18, count)
        if visible_count <= 0:
            return
        dot_radius = max(1, int(radius * 0.075))
        for index in range(visible_count):
            angle = index * (np.pi * 2 / max(1, visible_count)) + ring_scale * 1.7
            wobble = 0.72 + 0.18 * ((index * 17) % 9) / 8.0
            dot_x = int(center[0] + np.cos(angle) * radius * ring_scale * wobble)
            dot_y = int(center[1] + np.sin(angle) * radius * ring_scale * wobble)
            pygame.draw.circle(surface, color, (dot_x, dot_y), dot_radius)

    def _draw_storage_bars(self, screen: pygame.Surface, sx: int, sy: int, radius: int) -> None:
        colony = self.engine.colony
        items = (
            (ResourceType.LEAVES, RESOURCE_SPECS[ResourceType.LEAVES].color),
            (ResourceType.WATER, RESOURCE_SPECS[ResourceType.WATER].color),
            (ResourceType.PROTEIN, RESOURCE_SPECS[ResourceType.PROTEIN].color),
            (ResourceType.WASTE, RESOURCE_SPECS[ResourceType.WASTE].color),
            (ResourceType.FOOD, FOOD_STORE),
            (ResourceType.FUNGUS, RESOURCE_SPECS[ResourceType.FUNGUS].color),
            (ResourceType.PROTEIN_PASTE, RESOURCE_SPECS[ResourceType.PROTEIN_PASTE].color),
            (ResourceType.COMPOST, RESOURCE_SPECS[ResourceType.COMPOST].color),
        )
        bar_width = max(34, int(radius * 1.65))
        bar_height = max(2, int(3 * max(1.0, self.engine.config.render.cell_size / 6)))
        x = sx - bar_width // 2
        y = sy + radius + 8
        total_height = len(items) * bar_height + (len(items) - 1) * 2
        pygame.draw.rect(screen, (5, 8, 8), (x - 1, y - 1, bar_width + 2, total_height + 2), border_radius=3)
        for index, (resource_type, color) in enumerate(items):
            row_y = y + index * (bar_height + 2)
            desired = max(1.0, colony.desired_storage_for(resource_type))
            fill = float(np.clip(colony.inventory.get(resource_type) / desired, 0.0, 1.0))
            pygame.draw.rect(screen, (39, 45, 38), (x, row_y, bar_width, bar_height), border_radius=2)
            pygame.draw.rect(screen, color, (x, row_y, int(bar_width * fill), bar_height), border_radius=2)

    def _draw_ants(
        self,
        screen: pygame.Surface,
        camera: Camera,
        selected_ant_id: int | None,
        profile: QualityProfile,
    ) -> None:
        margin = 12
        panel_left = screen.get_width() - self.engine.config.render.ui_width
        visible: list[tuple[Ant, int, int]] = []
        selected: tuple[Ant, int, int] | None = None

        for ant in self.engine.ants:
            sx, sy = camera.world_to_screen(ant.x, ant.y)
            if sx < -margin or sy < -margin or sx > panel_left + margin or sy > screen.get_height() + margin:
                continue
            if ant.ant_id == selected_ant_id:
                selected = (ant, sx, sy)
            else:
                visible.append((ant, sx, sy))

        lod_context = AntLODContext(
            visible_count=len(visible),
            zoom=camera.zoom,
            config=self.engine.config.render,
        )
        draw_stride = max(1, profile.ant_stride)
        for index, (ant, sx, sy) in enumerate(visible):
            if draw_stride > 1 and index % draw_stride != 0:
                continue
            lod = lod_context.lod_for_ant(False)
            if lod == AntRenderLOD.MID and len(visible) > profile.max_mid_ants:
                lod = AntRenderLOD.FAR
            self._draw_ant(screen, camera, ant, sx, sy, lod, selected=False)

        if selected is not None:
            ant, sx, sy = selected
            self._draw_ant(screen, camera, ant, sx, sy, AntRenderLOD.CLOSE, selected=True)

    def _draw_ant(
        self,
        screen: pygame.Surface,
        camera: Camera,
        ant: Ant,
        sx: int,
        sy: int,
        lod: AntRenderLOD,
        selected: bool,
    ) -> None:
        if lod == AntRenderLOD.FAR:
            self._draw_far_ant(screen, camera, ant, sx, sy)
        elif lod == AntRenderLOD.MID:
            self._draw_mid_ant(screen, camera, ant, sx, sy)
        else:
            self._draw_close_ant(screen, camera, ant, sx, sy, selected)

    def _draw_far_ant(self, screen: pygame.Surface, camera: Camera, ant: Ant, sx: int, sy: int) -> None:
        color = self.engine.colony.color
        if self.overlays.render_quality != "fast":
            tail = max(2, int(2.6 * camera.zoom))
            start = (int(sx - np.cos(ant.angle) * tail), int(sy - np.sin(ant.angle) * tail))
            pygame.draw.line(screen, (22, 96, 112), start, (sx, sy), 1)
        pygame.draw.circle(screen, color, (sx, sy), 1)
        if ant.has_cargo:
            pygame.draw.circle(screen, _cargo_color(ant), (sx, sy), 1)

    def _draw_mid_ant(self, screen: pygame.Surface, camera: Camera, ant: Ant, sx: int, sy: int) -> None:
        base_radius = 1.5 + 0.75 * camera.zoom
        if ant.inferred_role == "Scout":
            base_radius += 0.4
        if ant.inferred_role == "Carrier":
            base_radius += 0.2
        if ant.inferred_role in ("Soldier", "Defender"):
            base_radius += 0.5
        if ant.inferred_role in ("Farmer", "Nurse", "Builder"):
            base_radius += 0.25
        radius = max(2, int(base_radius))
        color = self.engine.colony.color
        pygame.draw.circle(screen, ANT_OUTLINE, (sx, sy), radius + 1)
        pygame.draw.circle(screen, color, (sx, sy), radius)
        if ant.has_cargo:
            cargo_x = int(sx + np.cos(ant.angle) * (radius + 2))
            cargo_y = int(sy + np.sin(ant.angle) * (radius + 2))
            glint = self.art.cargo_glint(max(2, radius), _cargo_color(ant))
            screen.blit(glint, (cargo_x - glint.get_width() // 2, cargo_y - glint.get_height() // 2), special_flags=pygame.BLEND_ADD)

    def _draw_close_ant(self, screen: pygame.Surface, camera: Camera, ant: Ant, sx: int, sy: int, selected: bool) -> None:
        radius = max(3, int(2.2 + 1.25 * camera.zoom))
        color = self.engine.colony.color
        outline = ANT_OUTLINE
        if selected:
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), radius * 4, 1)

        abdomen, thorax, head = self._ant_body_points(sx, sy, ant.angle, radius)
        leg_color = (20, 48, 54)
        if radius >= 4:
            for offset in (-0.65, 0.0, 0.65):
                lx = int(thorax[0] - np.sin(ant.angle) * radius * offset)
                ly = int(thorax[1] + np.cos(ant.angle) * radius * offset)
                for side in (-1, 1):
                    ex = int(lx + np.cos(ant.angle + side * 1.25) * radius * 1.6)
                    ey = int(ly + np.sin(ant.angle + side * 1.25) * radius * 1.6)
                    pygame.draw.line(screen, leg_color, (lx, ly), (ex, ey), 1)

        pygame.draw.circle(screen, outline, abdomen, int(radius * 1.35) + 1)
        pygame.draw.circle(screen, tuple(max(0, min(255, int(c * 0.72))) for c in color), abdomen, int(radius * 1.35))
        pygame.draw.circle(screen, outline, thorax, radius + 1)
        pygame.draw.circle(screen, color, thorax, radius)
        pygame.draw.circle(screen, outline, head, max(2, int(radius * 0.72)) + 1)
        pygame.draw.circle(screen, (94, 235, 255), head, max(2, int(radius * 0.72)))
        if ant.has_cargo:
            cargo_x = int(head[0] + np.cos(ant.angle) * radius * 1.45)
            cargo_y = int(head[1] + np.sin(ant.angle) * radius * 1.45)
            glint = self.art.cargo_glint(max(3, int(radius * 0.85)), _cargo_color(ant))
            screen.blit(glint, (cargo_x - glint.get_width() // 2, cargo_y - glint.get_height() // 2), special_flags=pygame.BLEND_ADD)

    def _ant_body_points(self, sx: int, sy: int, angle: float, radius: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        dx = np.cos(angle)
        dy = np.sin(angle)
        abdomen = (int(sx - dx * radius * 1.25), int(sy - dy * radius * 1.25))
        thorax = (sx, sy)
        head = (int(sx + dx * radius * 1.25), int(sy + dy * radius * 1.25))
        return abdomen, thorax, head


def _scale_surface(surface: pygame.Surface, size: tuple[int, int], *, smooth: bool) -> pygame.Surface:
    if surface.get_size() == size:
        return surface
    if smooth:
        return pygame.transform.smoothscale(surface, size)
    return pygame.transform.scale(surface, size)


def _smooth_timings(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    if not previous:
        return current
    smoothed: dict[str, float] = {}
    for key, value in current.items():
        old = previous.get(key, value)
        smoothed[key] = old * 0.82 + value * 0.18
    return smoothed


_RESOURCE_PHEROMONE_TYPES = frozenset(RESOURCE_PHEROMONES.values())


def _cargo_color(ant: Ant) -> tuple[int, int, int]:
    if ant.inventory_type is None:
        return CARGO_LEAF
    return _resource_color(ant.inventory_type)


def _resource_color(resource_type: ResourceType) -> tuple[int, int, int]:
    if resource_type == ResourceType.FOOD:
        return FOOD_STORE
    return RESOURCE_SPECS[resource_type].color
