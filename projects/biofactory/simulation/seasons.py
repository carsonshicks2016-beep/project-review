from __future__ import annotations

from dataclasses import dataclass


SEASON_NAMES = ("Spring", "Summer", "Autumn", "Winter")


@dataclass
class SeasonSystem:
    day_length_ticks: int = 1800
    season_length_days: int = 12
    season_index: int = 0
    day: int = 1

    @property
    def name(self) -> str:
        return SEASON_NAMES[self.season_index]

    @property
    def temperature_delta(self) -> float:
        return (0.00, 0.08, -0.02, -0.12)[self.season_index]

    @property
    def humidity_delta(self) -> float:
        return (0.04, -0.03, 0.02, -0.05)[self.season_index]

    @property
    def light_multiplier(self) -> float:
        return (1.00, 1.05, 0.92, 0.78)[self.season_index]

    def update(self, tick: int) -> None:
        if tick > 0 and tick % self.day_length_ticks == 0:
            self.day += 1
            if self.day % self.season_length_days == 0:
                self.season_index = (self.season_index + 1) % len(SEASON_NAMES)
