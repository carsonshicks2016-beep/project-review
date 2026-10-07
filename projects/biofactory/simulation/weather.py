from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeatherSystem:
    state: str = "Clear"
    intensity: float = 0.0

    @property
    def temperature_delta(self) -> float:
        return 0.0

    @property
    def humidity_delta(self) -> float:
        return 0.0

    @property
    def light_multiplier(self) -> float:
        return 1.0

    @property
    def movement_multiplier(self) -> float:
        return 1.0

    @property
    def pheromone_decay_multiplier(self) -> float:
        return 1.0

    @property
    def pheromone_diffusion_multiplier(self) -> float:
        return 1.0

    def update(self, tick: int) -> None:
        # MVP keeps weather inert. The hook is here so pheromones and terrain can
        # receive weather modifiers when rain, wind, drought, and floods arrive.
        if tick % 3600 == 0:
            self.state = "Clear"
            self.intensity = 0.0
