"""
High-performance Level-2 Limit Order Book representation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


class OrderBook:
    """
    Level-2 Limit Order Book with fast numpy vectorization.
    Maintains sorted price levels for bids (descending) and asks (ascending).
    """

    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Internal price -> size maps
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        
        # Cached sorted arrays for vectorized computation
        self._bids_cache: Optional[np.ndarray] = None
        self._asks_cache: Optional[np.ndarray] = None
        self._dirty: bool = True

    def clear(self):
        self._bids.clear()
        self._asks.clear()
        self._bids_cache = None
        self._asks_cache = None
        self._dirty = True

    def update_bid(self, price: float, size: float):
        p = round(round(price / self.tick_size) * self.tick_size, 8)
        if size <= 1e-8:
            self._bids.pop(p, None)
        else:
            self._bids[p] = float(size)
        self._dirty = True

    def update_ask(self, price: float, size: float):
        p = round(round(price / self.tick_size) * self.tick_size, 8)
        if size <= 1e-8:
            self._asks.pop(p, None)
        else:
            self._asks[p] = float(size)
        self._dirty = True

    def set_snapshot(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Set entire orderbook state from snapshot list of (price, size)."""
        self._bids.clear()
        self._asks.clear()
        for p, s in bids:
            if s > 1e-8:
                self._bids[round(round(p / self.tick_size) * self.tick_size, 8)] = float(s)
        for p, s in asks:
            if s > 1e-8:
                self._asks[round(round(p / self.tick_size) * self.tick_size, 8)] = float(s)
        self._dirty = True

    def _rebuild_cache(self):
        if not self._dirty:
            return

        # Sort bids descending
        if len(self._bids) > 0:
            sorted_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)
            self._bids_cache = np.array(sorted_bids, dtype=np.float64)
        else:
            self._bids_cache = np.empty((0, 2), dtype=np.float64)

        # Sort asks ascending
        if len(self._asks) > 0:
            sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])
            self._asks_cache = np.array(sorted_asks, dtype=np.float64)
        else:
            self._asks_cache = np.empty((0, 2), dtype=np.float64)

        self._dirty = False

    @property
    def bids(self) -> np.ndarray:
        self._rebuild_cache()
        return self._bids_cache

    @property
    def asks(self) -> np.ndarray:
        self._rebuild_cache()
        return self._asks_cache

    @property
    def best_bid(self) -> Tuple[float, float]:
        b = self.bids
        if len(b) == 0:
            return (0.0, 0.0)
        return (b[0, 0], b[0, 1])

    @property
    def best_ask(self) -> Tuple[float, float]:
        a = self.asks
        if len(a) == 0:
            return (float("inf"), 0.0)
        return (a[0, 0], a[0, 1])

    @property
    def mid_price(self) -> float:
        bb, _ = self.best_bid
        ba, _ = self.best_ask
        if bb > 0 and ba < float("inf"):
            return 0.5 * (bb + ba)
        return 0.0

    @property
    def spread(self) -> float:
        bb, _ = self.best_bid
        ba, _ = self.best_ask
        if bb > 0 and ba < float("inf"):
            return max(0.0, ba - bb)
        return 0.0

    def get_depth_slice(self, depth: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Returns top `depth` levels of bids and asks."""
        b = self.bids[:depth]
        a = self.asks[:depth]
        return b, a
