"""Unit tests for Level-2 OrderBook."""

import pytest
import numpy as np
from src.core.orderbook import OrderBook


def test_orderbook_bids_asks_sorting():
    ob = OrderBook(symbol="BTC/USD", tick_size=0.50)
    ob.update_bid(100.0, 1.5)
    ob.update_bid(101.5, 2.0)
    ob.update_bid(99.0, 0.5)

    ob.update_ask(103.0, 1.0)
    ob.update_ask(102.5, 3.0)
    ob.update_ask(105.0, 0.8)

    # Best bid should be highest (101.5)
    assert ob.best_bid == (101.5, 2.0)
    # Best ask should be lowest (102.5)
    assert ob.best_ask == (102.5, 3.0)
    assert ob.mid_price == 102.0
    assert ob.spread == 1.0


def test_orderbook_deletion():
    ob = OrderBook(symbol="BTC/USD", tick_size=0.50)
    ob.update_bid(100.0, 1.0)
    assert ob.best_bid == (100.0, 1.0)

    # Deletion via 0 size
    ob.update_bid(100.0, 0.0)
    assert ob.best_bid == (0.0, 0.0)
