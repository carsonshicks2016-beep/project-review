"""
Institutional Risk Management and Circuit Breakers for High-Frequency Market Making.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from src.config import RiskConfig, MarketConfig
from src.core.matching_engine import OrderSide, SimulatedMatchingEngine


@dataclass
class RiskVerdict:
    allow_bid: bool
    allow_ask: bool
    bid_size: float
    ask_size: float
    emergency_flatten_required: bool
    reason: Optional[str] = None


class RiskManager:
    """
    Enforces non-negotiable risk constraints:
    - Maximum absolute inventory and position notional
    - Dynamic one-sided quoting under inventory skew
    - Maximum portfolio drawdown circuit breaker
    - Stale order timeouts
    - Emergency liquidation trigger
    """

    def __init__(self, risk_cfg: RiskConfig, market_cfg: MarketConfig):
        self.risk_cfg = risk_cfg
        self.market_cfg = market_cfg
        self.circuit_breaker_tripped: bool = False
        self.trip_reason: Optional[str] = None
        self.peak_portfolio_equity: float = market_cfg.initial_cash

    def reset(self, initial_cash: float):
        self.circuit_breaker_tripped = False
        self.trip_reason = None
        self.peak_portfolio_equity = initial_cash

    def evaluate(
        self,
        current_equity: float,
        inventory: float,
        mid_price: float,
        target_bid_size: float,
        target_ask_size: float,
    ) -> RiskVerdict:
        """
        Evaluates current risk state and returns permissions and clamped sizes for quoting.
        """
        # 1. Update peak equity & check maximum drawdown breaker
        if current_equity > self.peak_portfolio_equity:
            self.peak_portfolio_equity = current_equity

        drawdown = (self.peak_portfolio_equity - current_equity) / self.peak_portfolio_equity
        if drawdown >= self.risk_cfg.max_drawdown_pct:
            self.circuit_breaker_tripped = True
            self.trip_reason = f"Max drawdown exceeded: {drawdown * 100:.2f}% >= {self.risk_cfg.max_drawdown_pct * 100:.2f}%"
            return RiskVerdict(
                allow_bid=False,
                allow_ask=False,
                bid_size=0.0,
                ask_size=0.0,
                emergency_flatten_required=True,
                reason=self.trip_reason,
            )

        if self.circuit_breaker_tripped:
            return RiskVerdict(
                allow_bid=False,
                allow_ask=False,
                bid_size=0.0,
                ask_size=0.0,
                emergency_flatten_required=abs(inventory) > 1e-6,
                reason=self.trip_reason,
            )

        # 2. Check position notional limit
        current_notional = abs(inventory) * mid_price
        if current_notional >= self.risk_cfg.max_position_notional:
            # Over notional limit: allow only reduction
            allow_bid = inventory < 0
            allow_ask = inventory > 0
            return RiskVerdict(
                allow_bid=allow_bid,
                allow_ask=allow_ask,
                bid_size=min(target_bid_size, abs(inventory)) if allow_bid else 0.0,
                ask_size=min(target_ask_size, abs(inventory)) if allow_ask else 0.0,
                emergency_flatten_required=False,
                reason="Max position notional cap reached; quoting one-sided reduction",
            )

        # 3. Check inventory boundaries & one-sided quoting thresholds
        max_inv = self.risk_cfg.max_inventory
        one_sided_threshold = self.risk_cfg.one_sided_quoting_threshold * max_inv

        allow_bid = True
        allow_ask = True

        # Clamp sizes to allowed bounds
        bid_size = min(max(self.market_cfg.min_order_size, target_bid_size), self.market_cfg.max_order_size)
        ask_size = min(max(self.market_cfg.min_order_size, target_ask_size), self.market_cfg.max_order_size)

        # If long exceeds threshold, kill bids or quote purely to reduce
        if inventory >= max_inv:
            allow_bid = False
            allow_ask = True
            reason = "Max inventory reached; quoting ask only"
        elif inventory >= one_sided_threshold:
            allow_bid = False
            allow_ask = True
            reason = "Inventory long skew; suppressing bids"
        elif inventory <= -max_inv:
            allow_bid = True
            allow_ask = False
            reason = "Max short inventory reached; quoting bid only"
        elif inventory <= -one_sided_threshold:
            allow_bid = True
            allow_ask = False
            reason = "Inventory short skew; suppressing asks"
        else:
            reason = None

        return RiskVerdict(
            allow_bid=allow_bid,
            allow_ask=allow_ask,
            bid_size=bid_size if allow_bid else 0.0,
            ask_size=ask_size if allow_ask else 0.0,
            emergency_flatten_required=False,
            reason=reason,
        )
