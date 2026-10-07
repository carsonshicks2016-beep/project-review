from __future__ import annotations

from dataclasses import dataclass

from biofactory.config import RenderConfig


@dataclass
class Camera:
    x: float
    y: float
    screen_width: int
    screen_height: int
    render_config: RenderConfig
    zoom: float = 1.0

    @property
    def scale(self) -> float:
        return self.render_config.cell_size * self.zoom

    def world_to_screen(self, world_x: float, world_y: float) -> tuple[int, int]:
        sx = (world_x - self.x) * self.scale + self.screen_width * 0.5
        sy = (world_y - self.y) * self.scale + self.screen_height * 0.5
        return int(sx), int(sy)

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        wx = (screen_x - self.screen_width * 0.5) / self.scale + self.x
        wy = (screen_y - self.screen_height * 0.5) / self.scale + self.y
        return wx, wy

    def pan_screen(self, dx: float, dy: float) -> None:
        self.x += dx / self.scale
        self.y += dy / self.scale

    def pan_world(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def zoom_at(self, factor: float, screen_x: float, screen_y: float) -> None:
        before = self.screen_to_world(screen_x, screen_y)
        self.zoom = max(self.render_config.min_zoom, min(self.render_config.max_zoom, self.zoom * factor))
        after = self.screen_to_world(screen_x, screen_y)
        self.x += before[0] - after[0]
        self.y += before[1] - after[1]

    def center_on(self, world_x: float, world_y: float) -> None:
        self.x = world_x
        self.y = world_y
