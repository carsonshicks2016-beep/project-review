"""Shared portfolio-state features for training, backtests, and live trading."""

import numpy as np


def normalized_unrealized_pnl(position_qty: float, entry_price: float, current_price: float) -> float:
    """Return the same normalized unrealized P&L feature used during training."""
    if position_qty <= 0 or entry_price <= 0:
        return 0.5

    raw_pnl = (float(current_price) - float(entry_price)) / (float(entry_price) + 1e-10)
    scaled = np.clip(10.0 * raw_pnl, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-scaled)))


def raw_unrealized_pnl(position_qty: float, entry_price: float, current_price: float) -> float:
    """Return raw percentage P&L for logs and operator-readable output."""
    if position_qty <= 0 or entry_price <= 0:
        return 0.0
    return float((float(current_price) - float(entry_price)) / (float(entry_price) + 1e-10))


def build_portfolio_features(position_qty: float, entry_price: float, current_price: float) -> np.ndarray:
    """Build [position_flag, normalized_unrealized_pnl] exactly once for all callers."""
    position_flag = 1.0 if position_qty > 0 else 0.0
    unrealized_norm = normalized_unrealized_pnl(position_qty, entry_price, current_price)
    return np.array([position_flag, unrealized_norm], dtype=np.float32)


def current_allocation(cash: float, position_qty: float, current_price: float) -> float:
    """Return current asset allocation as a fraction of total portfolio value."""
    asset_value = float(position_qty) * float(current_price)
    portfolio_value = float(cash) + asset_value
    if portfolio_value <= 0:
        return 0.0
    return float(np.clip(asset_value / portfolio_value, 0.0, 1.0))
