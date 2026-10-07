from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from biofactory.pheromones.diffusion import diffuse_and_decay
from biofactory.pheromones.pheromone_config import PHEROMONE_SPECS, PheromoneType
from biofactory.simulation.world import World


@dataclass
class PheromoneLayers:
    width: int
    height: int
    layers: dict[PheromoneType, np.ndarray]

    @classmethod
    def create(cls, width: int, height: int) -> "PheromoneLayers":
        layers = {
            pheromone_type: np.zeros((height, width), dtype=np.float32)
            for pheromone_type in PHEROMONE_SPECS
        }
        return cls(width=width, height=height, layers=layers)

    def add_at(self, pheromone_type: PheromoneType, x: float, y: float, amount: float, radius: int = 0) -> None:
        layer = self.layers[pheromone_type]
        cx, cy = self._cell(x, y)
        if radius <= 0:
            layer[cy, cx] += amount
        else:
            y0 = max(0, cy - radius)
            y1 = min(self.height, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(self.width, cx + radius + 1)
            layer[y0:y1, x0:x1] += amount / max(1, (y1 - y0) * (x1 - x0)) * 4.0

    def sample(self, pheromone_type: PheromoneType, x: float, y: float) -> float:
        layer = self.layers[pheromone_type]
        cx, cy = self._cell(x, y)
        return float(layer[cy, cx])

    def sample_direction(self, pheromone_type: PheromoneType, x: float, y: float, angle: float, distance: float) -> float:
        sx = x + math.cos(angle) * distance
        sy = y + math.sin(angle) * distance
        return self.sample(pheromone_type, sx, sy)

    def gradient(self, pheromone_type: PheromoneType, x: float, y: float) -> tuple[float, float]:
        layer = self.layers[pheromone_type]
        cx, cy = self._cell(x, y)
        left = layer[cy, max(0, cx - 1)]
        right = layer[cy, min(self.width - 1, cx + 1)]
        up = layer[max(0, cy - 1), cx]
        down = layer[min(self.height - 1, cy + 1), cx]
        return float(right - left), float(down - up)

    def update(self, world: World) -> None:
        for pheromone_type, layer in self.layers.items():
            if float(np.max(layer)) <= 1e-6:
                continue
            spec = PHEROMONE_SPECS[pheromone_type]
            self.layers[pheromone_type] = diffuse_and_decay(
                layer,
                diffusion_rate=spec.diffusion_rate,
                decay_rate=spec.decay_rate,
                max_concentration=spec.max_concentration,
                terrain_decay_multiplier=world.pheromone_decay_multiplier,
                terrain_diffusion_multiplier=world.pheromone_diffusion_multiplier,
            )

    def max_value(self, pheromone_type: PheromoneType) -> float:
        return float(np.max(self.layers[pheromone_type]))

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = min(self.width - 1, max(0, int(x)))
        cy = min(self.height - 1, max(0, int(y)))
        return cx, cy
