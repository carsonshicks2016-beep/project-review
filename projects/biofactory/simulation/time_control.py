from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeControl:
    speed_options: tuple[int, ...]
    speed_index: int = 0
    paused: bool = False
    step_once: bool = False

    @property
    def speed(self) -> int:
        return self.speed_options[self.speed_index]

    def set_speed_by_value(self, value: int) -> None:
        if value in self.speed_options:
            self.speed_index = self.speed_options.index(value)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def request_step(self) -> None:
        self.step_once = True

    def consume_step_request(self) -> bool:
        if self.step_once:
            self.step_once = False
            return True
        return False
