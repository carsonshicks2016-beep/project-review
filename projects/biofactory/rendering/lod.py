from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from biofactory.config import RenderConfig


class AntRenderLOD(str, Enum):
    FAR = "far"
    MID = "mid"
    CLOSE = "close"


@dataclass(frozen=True)
class AntLODContext:
    visible_count: int
    zoom: float
    config: RenderConfig

    def lod_for_ant(self, selected: bool = False) -> AntRenderLOD:
        if selected:
            return AntRenderLOD.CLOSE
        if self.zoom >= self.config.close_ant_zoom and self.visible_count <= self.config.max_detailed_ants:
            return AntRenderLOD.CLOSE
        if self.zoom >= self.config.mid_ant_zoom:
            return AntRenderLOD.MID
        return AntRenderLOD.FAR
