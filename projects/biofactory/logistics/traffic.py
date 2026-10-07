from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrafficMetrics:
    congested_cells: int = 0
    max_traffic: float = 0.0
    average_traffic: float = 0.0

    @classmethod
    def from_layer(cls, traffic_layer: np.ndarray) -> "TrafficMetrics":
        return cls(
            congested_cells=int(np.count_nonzero(traffic_layer > 22.0)),
            max_traffic=float(np.max(traffic_layer)),
            average_traffic=float(np.mean(traffic_layer)),
        )
