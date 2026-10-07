"""Unit tests for SimulatedMatchingEngine (Queue priority, partial fills, maker rebates)."""

import pytest
from src.core.matching_engine import SimulatedMatchingEngine, OrderSide


def test_queue_priority_and_fill():
    engine = SimulatedMatchingEngine(
        symbol="BTC/USD",
        maker_fee=-0.0001, # Negative fee = 1 bps rebate
        initial_cash=10_000.0,
        latency_sec=0.0,
    )

    # Place bid at $100 with 1.0 size and 2.0 ahead in queue
    engine.place_limit_order(
        order_id="TEST_BID",
        side=OrderSide.BUY,
        price=100.0,
        size=1.0,
        current_book_depth_at_price=2.0,
        timestamp=0.0,
    )

    # Small trade of 1.0 at $100: should NOT fill our order because 2.0 is ahead!
    fills1 = engine.process_trade(trade_price=100.0, trade_size=1.0, is_buyer_maker=True, timestamp=0.1)
    assert len(fills1) == 0
    assert engine.active_orders["TEST_BID"].queue_ahead == 1.0

    # Another trade of 1.5 at $100: consumes remaining 1.0 queue ahead and fills 0.5 of our order!
    fills2 = engine.process_trade(trade_price=100.0, trade_size=1.5, is_buyer_maker=True, timestamp=0.2)
    assert len(fills2) == 1
    assert fills2[0].size == 0.5
    assert engine.active_orders["TEST_BID"].filled_size == 0.5
    assert engine.inventory == 0.5
    assert engine.total_maker_rebates > 0.0 # Earned rebate!


def test_adverse_selection_sweep():
    engine = SimulatedMatchingEngine(
        symbol="BTC/USD",
        maker_fee=-0.0001,
        initial_cash=10_000.0,
        latency_sec=0.0,
    )

    engine.place_limit_order(
        order_id="TEST_BID",
        side=OrderSide.BUY,
        price=100.0,
        size=0.1,
        current_book_depth_at_price=0.0,
        timestamp=0.0,
    )

    # Big sell sweep drops down through 100 to 98
    fills = engine.process_trade(trade_price=98.0, trade_size=5.0, is_buyer_maker=True, timestamp=0.1)
    assert len(fills) == 1
    assert fills[0].price == 100.0
    assert "TEST_BID" not in engine.active_orders
