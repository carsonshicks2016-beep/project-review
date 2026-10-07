from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Iterable


@dataclass
class SpatialHash:
    cell_size: float
    buckets: DefaultDict[tuple[int, int], list[int]] = field(default_factory=lambda: defaultdict(list))

    def clear(self) -> None:
        self.buckets.clear()

    def insert(self, entity_id: int, x: float, y: float) -> None:
        self.buckets[self._key(x, y)].append(entity_id)

    def nearby_ids(self, x: float, y: float, radius_cells: int = 1) -> Iterable[int]:
        cx, cy = self._key(x, y)
        for by in range(cy - radius_cells, cy + radius_cells + 1):
            for bx in range(cx - radius_cells, cx + radius_cells + 1):
                yield from self.buckets.get((bx, by), ())

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self.cell_size), int(y // self.cell_size)
