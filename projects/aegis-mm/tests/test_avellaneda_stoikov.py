"""Unit tests for Avellaneda-Stoikov mathematical calculations."""

import pytest
from src.math.avellaneda_stoikov import AvellanedaStoikov


def test_as_reservation_price_inventory_skew():
    as_model = AvellanedaStoikov(gamma=0.1, sigma=0.02, kappa=1.5, time_horizon=60.0, tick_size=0.1)
    mid = 100.0

    # Flat inventory: reservation price equals mid price
    r_flat = as_model.reservation_price(mid, inventory=0.0, time_left=30.0)
    assert abs(r_flat - mid) < 1e-6

    # Long inventory (q > 0): reservation price must be strictly LOWER than mid to encourage selling
    r_long = as_model.reservation_price(mid, inventory=10.0, time_left=30.0)
    assert r_long < mid

    # Short inventory (q < 0): reservation price must be strictly HIGHER than mid to encourage buying
    r_short = as_model.reservation_price(mid, inventory=-10.0, time_left=30.0)
    assert r_short > mid


def test_as_optimal_quotes_no_cross():
    as_model = AvellanedaStoikov(gamma=0.05, sigma=0.01, kappa=1.5, tick_size=0.50)
    mid = 1000.0

    quotes = as_model.compute_quotes(mid, inventory=2.0)
    assert quotes.bid_price < mid
    assert quotes.ask_price > mid
    assert quotes.bid_price < quotes.ask_price
    assert quotes.total_spread >= 1.0  # At least 2 ticks
