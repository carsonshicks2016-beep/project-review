from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from biofactory.pheromones.pheromone_config import PheromoneType

VisualMode = Literal["hybrid", "circuit", "natural"]
VISUAL_MODES: tuple[VisualMode, ...] = ("hybrid", "circuit", "natural")
RenderQuality = Literal["fast", "balanced", "quality"]
RENDER_QUALITIES: tuple[RenderQuality, ...] = ("fast", "balanced", "quality")


@dataclass
class OverlayState:
    visual_mode: VisualMode = "hybrid"
    render_quality: RenderQuality = "balanced"
    pheromones: dict[PheromoneType, bool] = field(
        default_factory=lambda: {
            PheromoneType.FOOD: True,
            PheromoneType.WATER: True,
            PheromoneType.PROTEIN: True,
            PheromoneType.WASTE: True,
            PheromoneType.DEMAND: True,
            PheromoneType.TRAFFIC: True,
        }
    )
    resources: bool = True
    traffic_heatmap: bool = False

    def toggle(self, pheromone_type: PheromoneType) -> None:
        self.pheromones[pheromone_type] = not self.pheromones.get(pheromone_type, False)

    def cycle_visual_mode(self) -> None:
        index = VISUAL_MODES.index(self.visual_mode)
        self.visual_mode = VISUAL_MODES[(index + 1) % len(VISUAL_MODES)]

    def cycle_render_quality(self) -> None:
        index = RENDER_QUALITIES.index(self.render_quality)
        self.render_quality = RENDER_QUALITIES[(index + 1) % len(RENDER_QUALITIES)]

    @property
    def circuit_mode(self) -> bool:
        return self.visual_mode == "circuit"
