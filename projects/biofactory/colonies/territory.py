from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TerritoryMap:
    values: np.ndarray

    @classmethod
    def create(cls, width: int, height: int) -> "TerritoryMap":
        return cls(values=np.zeros((height, width), dtype=np.float32))

    def mark(self, x: float, y: float, amount: float) -> None:
        cx = min(self.values.shape[1] - 1, max(0, int(x)))
        cy = min(self.values.shape[0] - 1, max(0, int(y)))
        self.values[cy, cx] = min(1.0, self.values[cy, cx] + amount)
