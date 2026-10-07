"""
High-frequency market microstructure metrics:
- Order Flow Imbalance (OFI) - Cont, Kukanov & Stoikov (2014)
- Micro-Price (Volume-Weighted Mid-Price) - Stoikov (2018)
- Multi-Level Book Depth Imbalance (VOI)
- Realized Volatility tracking
- Order Flow Toxicity proxy
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import math
import numpy as np


@dataclass
class MicrostructureFeatures:
    mid_price: float
    micro_price: float
    spread: float
    spread_bps: float
    depth_imbalance: float          # In [-1.0, 1.0]
    multi_level_imbalance: float    # Top-K level imbalance
    ofi: float                      # Order flow imbalance
    realized_volatility: float      # Rolling realized vol
    trade_imbalance: float          # Rolling buy vs sell trade flow
    toxicity_score: float           # Proxy for adverse selection risk [0.0, 1.0]


class MicrostructureEngine:
    """
    Computes real-time streaming microstructure alpha features from L2 book updates and trades.
    """

    def __init__(self, window_size: int = 50, vol_decay: float = 0.94):
        self.window_size = window_size
        self.vol_decay = vol_decay
        
        # Historical state for OFI computation
        self.prev_best_bid_price: Optional[float] = None
        self.prev_best_bid_size: Optional[float] = None
        self.prev_best_ask_price: Optional[float] = None
        self.prev_best_ask_size: Optional[float] = None
        
        # Rolling buffers
        self.price_returns: Deque[float] = deque(maxlen=window_size)
        self.ofi_history: Deque[float] = deque(maxlen=window_size)
        self.trade_signs: Deque[float] = deque(maxlen=window_size)
        
        # Online variance
        self.ewma_vol: float = 0.001
        self.last_mid: Optional[float] = None

    def compute_ofi(
        self,
        bid_p: float,
        bid_s: float,
        ask_p: float,
        ask_s: float,
    ) -> float:
        """
        Cont-Kukanov-Stoikov Order Flow Imbalance:
        I_b = Q_b if P_b > P_b_prev; Q_b - Q_b_prev if P_b == P_b_prev; 0 if P_b < P_b_prev
        I_a = 0 if P_a > P_a_prev; Q_a - Q_a_prev if P_a == P_a_prev; Q_a if P_a < P_a_prev
        OFI = I_b - I_a
        """
        if self.prev_best_bid_price is None:
            self.prev_best_bid_price = bid_p
            self.prev_best_bid_size = bid_s
            self.prev_best_ask_price = ask_p
            self.prev_best_ask_size = ask_s
            return 0.0

        # Bid side contribution
        if bid_p > self.prev_best_bid_price:
            i_b = bid_s
        elif bid_p == self.prev_best_bid_price:
            i_b = bid_s - self.prev_best_bid_size
        else:
            i_b = -self.prev_best_bid_size

        # Ask side contribution
        if ask_p < self.prev_best_ask_price:
            i_a = ask_s
        elif ask_p == self.prev_best_ask_price:
            i_a = ask_s - self.prev_best_ask_size
        else:
            i_a = -self.prev_best_ask_size

        ofi = i_b - i_a

        # Update cache
        self.prev_best_bid_price = bid_p
        self.prev_best_bid_size = bid_s
        self.prev_best_ask_price = ask_p
        self.prev_best_ask_size = ask_s
        
        self.ofi_history.append(ofi)
        return ofi

    def update_volatility(self, current_mid: float) -> float:
        """Updates exponentially weighted realized volatility."""
        if self.last_mid is not None and self.last_mid > 0:
            ret = math.log(current_mid / self.last_mid)
            self.price_returns.append(ret)
            # EWMA update of return variance
            self.ewma_vol = math.sqrt(
                self.vol_decay * (self.ewma_vol ** 2) + (1.0 - self.vol_decay) * (ret ** 2)
            )
        self.last_mid = current_mid
        return max(1e-5, self.ewma_vol)

    def record_trade(self, size: float, is_buyer_maker: bool):
        """
        Records trade flow direction:
        If buyer was maker, taker was seller (-1.0).
        If seller was maker, taker was buyer (+1.0).
        """
        signed_vol = -size if is_buyer_maker else size
        self.trade_signs.append(signed_vol)

    def process(
        self,
        bids: np.ndarray,  # shape: (N, 2) [price, size]
        asks: np.ndarray,  # shape: (N, 2) [price, size]
    ) -> MicrostructureFeatures:
        """
        Processes current L2 book and produces full microstructure feature set.
        """
        best_bid_p, best_bid_s = bids[0, 0], bids[0, 1]
        best_ask_p, best_ask_s = asks[0, 0], asks[0, 1]
        
        mid = 0.5 * (best_bid_p + best_ask_p)
        spread = best_ask_p - best_bid_p
        spread_bps = (spread / mid) * 10_000.0 if mid > 0 else 0.0

        # Micro-price (Stoikov 2018):
        total_top_depth = best_bid_s + best_ask_s
        if total_top_depth > 0:
            micro_price = (best_bid_s * best_ask_p + best_ask_s * best_bid_p) / total_top_depth
            depth_imbalance = (best_bid_s - best_ask_s) / total_top_depth
        else:
            micro_price = mid
            depth_imbalance = 0.0

        # Multi-level imbalance across top K levels (e.g. 5)
        k = min(len(bids), len(asks), 5)
        k_bid_vol = np.sum(bids[:k, 1])
        k_ask_vol = np.sum(asks[:k, 1])
        k_total = k_bid_vol + k_ask_vol
        multi_level_imb = (k_bid_vol - k_ask_vol) / k_total if k_total > 0 else 0.0

        # OFI
        ofi = self.compute_ofi(best_bid_p, best_bid_s, best_ask_p, best_ask_s)

        # Volatility
        vol = self.update_volatility(mid)

        # Trade flow imbalance
        if len(self.trade_signs) > 0:
            total_vol = sum(abs(v) for v in self.trade_signs)
            net_vol = sum(self.trade_signs)
            trade_imbalance = net_vol / total_vol if total_vol > 0 else 0.0
        else:
            trade_imbalance = 0.0

        # Toxicity score proxy: high when OFI & trade flow agree and spread is widening
        ofi_norm = math.tanh(ofi / (total_top_depth + 1e-5))
        toxicity_score = min(1.0, max(0.0, 0.5 * abs(depth_imbalance) + 0.5 * abs(trade_imbalance)))

        return MicrostructureFeatures(
            mid_price=mid,
            micro_price=micro_price,
            spread=spread,
            spread_bps=spread_bps,
            depth_imbalance=depth_imbalance,
            multi_level_imbalance=multi_level_imb,
            ofi=ofi_norm,
            realized_volatility=vol,
            trade_imbalance=trade_imbalance,
            toxicity_score=toxicity_score,
        )
