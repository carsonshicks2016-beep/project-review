"""Unit tests for RiskManager circuit breakers and one-sided quoting."""

import pytest
from src.config import RiskConfig, MarketConfig
from src.risk.risk_manager import RiskManager


def test_one_sided_quoting_on_skew():
    risk_cfg = RiskConfig(max_inventory=1.0, one_sided_quoting_threshold=0.7)
    market_cfg = MarketConfig(initial_cash=10_000.0)
    rm = RiskManager(risk_cfg, market_cfg)

    # Balanced inventory (0.2) -> Allow both sides
    v1 = rm.evaluate(10_000.0, inventory=0.2, mid_price=100.0, target_bid_size=0.1, target_ask_size=0.1)
    assert v1.allow_bid is True
    assert v1.allow_ask is True

    # High Long Inventory (0.8 >= 0.70 * 1.0) -> Disallow bids, allow asks only
    v2 = rm.evaluate(10_000.0, inventory=0.8, mid_price=100.0, target_bid_size=0.1, target_ask_size=0.1)
    assert v2.allow_bid is False
    assert v2.allow_ask is True

    # High Short Inventory (-0.85) -> Disallow asks, allow bids only
    v3 = rm.evaluate(10_000.0, inventory=-0.85, mid_price=100.0, target_bid_size=0.1, target_ask_size=0.1)
    assert v3.allow_bid is True
    assert v3.allow_ask is False


def test_max_drawdown_circuit_breaker():
    risk_cfg = RiskConfig(max_drawdown_pct=0.05) # 5% max dd
    market_cfg = MarketConfig(initial_cash=10_000.0)
    rm = RiskManager(risk_cfg, market_cfg)

    # Equity drops from 10k to 9.4k (6% drawdown)
    v = rm.evaluate(9_400.0, inventory=0.5, mid_price=100.0, target_bid_size=0.1, target_ask_size=0.1)
    assert v.emergency_flatten_required is True
    assert rm.circuit_breaker_tripped is True
