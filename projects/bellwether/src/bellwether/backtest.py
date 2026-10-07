"""Phase 4 — Backtest Engine & Slippage Model.

Applies point-in-time discipline and a punishing micro-cap slippage model.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from . import db
from .logging_setup import get_logger

log = get_logger("bellwether.backtest")

def get_next_trading_open(conn, ticker: str, acceptance_datetime: str) -> dict[str, Any] | None:
    """Point-in-time discipline: find the exact daily bar where we'd fill.
    
    If the signal arrived at 10:00 AM on 2026-06-20, we still only use the 
    OPEN of the NEXT trading day (2026-06-21) to be brutally conservative.
    Never use the signal-bar close.
    """
    # Convert acceptance_datetime to date
    # Format is usually: 2026-06-22T07:55:04.000Z
    # We strip the time and add 1 day to find the strictly strictly > date.
    
    acc_dt = dt.datetime.fromisoformat(acceptance_datetime.replace('Z', '+00:00'))
    
    # We want the first price row where date > acc_dt.date()
    date_str = acc_dt.strftime("%Y-%m-%d")
    
    row = conn.execute(
        """
        SELECT * FROM prices 
        WHERE ticker = ? AND date > ? 
        ORDER BY date ASC 
        LIMIT 1
        """,
        (ticker, date_str)
    ).fetchone()
    
    if not row:
        return None
    return dict(row)

def get_future_price(conn, ticker: str, start_date: str, offset_days: int) -> dict[str, Any] | None:
    """Get the price row exactly N trading days after the start_date."""
    row = conn.execute(
        """
        SELECT * FROM prices 
        WHERE ticker = ? AND date >= ? 
        ORDER BY date ASC 
        LIMIT 1 OFFSET ?
        """,
        (ticker, start_date, offset_days)
    ).fetchone()
    
    if not row:
        return None
    return dict(row)

def calculate_slippage(fill_bar: dict[str, Any], adv_usd: float) -> float:
    """Calculate the estimated slippage penalty as a percentage.
    
    Implements a simplified proxy for Corwin-Schultz half-spread and 
    a square-root market impact term.
    """
    O, H, L, C = fill_bar["open"], fill_bar["high"], fill_bar["low"], fill_bar["close"]
    
    # 1. Spread estimator (High-Low Proxy)
    # The wider the intraday range, the wider the spread usually is.
    mid = (H + L) / 2
    if mid <= 0:
        return 0.05 # 5% default if data is weird
        
    daily_range_pct = (H - L) / mid
    
    # Assume the effective half-spread is ~10% of the daily range
    half_spread_pct = daily_range_pct * 0.10
    
    # 2. Square-Root Impact Term: ∝ σ·√(size/ADV)
    # Let's assume our trade size is fixed at $50,000 for this backtest
    trade_size = 50000.0
    adv = max(adv_usd, 1.0) # prevent div/0
    
    # Volatility proxy (sigma) is the daily range pct
    sigma = daily_range_pct
    
    # Impact = 0.1 * sigma * sqrt(trade_size / ADV)
    impact_pct = 0.1 * sigma * math.sqrt(trade_size / adv)
    
    # 3. Penalize thinly traded names heavily
    multiplier = 1.0
    if adv < 2_000_000:
        multiplier = 5.0
    elif adv < 5_000_000:
        multiplier = 3.0
        
    total_slippage = (half_spread_pct + impact_pct) * multiplier
    
    # Cap slippage at 15% (if it's that bad, the trade is untradable anyway)
    return min(total_slippage, 0.15)

def run_backtest(conn) -> None:
    """Iterate through signals, apply slippage, and calculate outcomes."""
    signals = conn.execute(
        """
        SELECT s.id as signal_id, s.direction, f.acceptance_datetime, c.ticker, c.adv_usd
        FROM signals s
        JOIN filings f ON s.filing_id = f.id
        JOIN companies c ON f.cik = c.cik
        WHERE s.id NOT IN (SELECT signal_id FROM outcomes)
        """
    ).fetchall()
    
    log.info("Running backtest for %d uncalculated signals...", len(signals))
    
    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        
        # Determine the benchmark based on crude sector logic
        benchmark = "XLE" # Defaulting to energy for our pond
        
        # 1. Point in Time Fill
        fill_bar = get_next_trading_open(conn, ticker, sig["acceptance_datetime"])
        if not fill_bar:
            log.warning("No future price data for %s after %s", ticker, sig["acceptance_datetime"])
            continue
            
        fill_date = fill_bar["date"]
        raw_fill_price = fill_bar["open"]
        
        # 2. Apply Slippage
        # If bullish, we buy (pay higher price). If bearish, we short (sell at lower price).
        slippage_pct = calculate_slippage(fill_bar, sig["adv_usd"])
        
        if direction == "bullish":
            realized_fill = raw_fill_price * (1 + slippage_pct)
        elif direction == "bearish":
            realized_fill = raw_fill_price * (1 - slippage_pct)
        else:
            realized_fill = raw_fill_price # Neutral trades don't incur slippage because we don't take them
            
        log.debug("Signal %s (%s): Raw open %.2f -> Realized fill %.2f (%.2f%% slippage)", 
                  sig["signal_id"], ticker, raw_fill_price, realized_fill, slippage_pct*100)
                  
        # 3. Calculate Forward Returns (+1d, +5d, +20d)
        fwd_returns = {}
        for window in [1, 5, 20]:
            target_bar = get_future_price(conn, ticker, fill_date, window)
            if target_bar:
                # Ret is (Target Close - Realized Fill) / Realized Fill
                ret = (target_bar["close"] - realized_fill) / realized_fill
                # If it's a bearish signal, a price drop is a positive return for us
                if direction == "bearish":
                    ret = -ret
                fwd_returns[f"{window}d"] = ret
            else:
                fwd_returns[f"{window}d"] = None
                
        # 4. Calculate Abnormal Return (vs Benchmark)
        abnormal_5d = None
        if fwd_returns["5d"] is not None:
            # Get benchmark return over exact same dates
            b_fill = conn.execute("SELECT open FROM prices WHERE ticker=? AND date=?", (benchmark, fill_date)).fetchone()
            target_bar_5d = get_future_price(conn, ticker, fill_date, 5) # Just to get the exact target date
            
            if b_fill and target_bar_5d:
                b_target = conn.execute("SELECT close FROM prices WHERE ticker=? AND date=?", (benchmark, target_bar_5d["date"])).fetchone()
                if b_target:
                    b_ret = (b_target["close"] - b_fill["open"]) / b_fill["open"]
                    if direction == "bearish":
                        b_ret = -b_ret
                    abnormal_5d = fwd_returns["5d"] - b_ret
                    
        # 5. Insert Outcome
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        conn.execute(
            """
            INSERT INTO outcomes (signal_id, fwd_ret_1d, fwd_ret_5d, fwd_ret_20d, abnormal_ret_5d, benchmark, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sig["signal_id"], fwd_returns.get("1d"), fwd_returns.get("5d"), fwd_returns.get("20d"), abnormal_5d, benchmark, now)
        )
        
    conn.commit()
    log.info("Backtest complete.")

if __name__ == "__main__":
    run_backtest(db.init_db())
