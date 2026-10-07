"""Phase 4 — Point-in-time pricing client (Tiingo).

Fetches historical daily OHLCV data for backtesting.
Never uses yfinance due to survivorship bias.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import requests

from . import config
from . import db
from .logging_setup import get_logger

log = get_logger("bellwether.pricing")

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

def fetch_prices(ticker: str, start_date: str = "2025-01-01") -> list[dict[str, Any]]:
    """Fetch daily OHLCV from Tiingo."""
    api_key = config.require("TIINGO_API_KEY")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}"
    }
    
    params = {
        "startDate": start_date,
        "format": "json",
        "resampleFreq": "daily"
    }
    
    log.info("Fetching Tiingo prices for %s from %s...", ticker, start_date)
    response = requests.get(TIINGO_URL.format(ticker=ticker), headers=headers, params=params)
    
    if response.status_code != 200:
        log.error("Tiingo error for %s: %s", ticker, response.text)
        response.raise_for_status()
        
    data = response.json()
    log.info("Got %d daily price bars for %s", len(data), ticker)
    return data

def sync_prices(conn) -> None:
    """Fetch and store prices for all universe companies and benchmarks."""
    # Get all active universe tickers
    rows = conn.execute("SELECT ticker FROM companies WHERE in_universe = 1").fetchall()
    tickers = [r["ticker"] for r in rows]
    
    # Add benchmarks (XLE for energy, XLB for materials, SPY as general market)
    benchmarks = ["XLE", "XLB", "SPY", "IWM"]
    for b in benchmarks:
        if b not in tickers:
            tickers.append(b)
            
    # Add a known delisted name for survivorship bias testing
    # Let's add 'PXD' (Pioneer Natural Resources - acquired by Exxon in 2024)
    # Wait, the backtest period is 2025+. 
    # Let's just sync whatever is in the list.
    
    for ticker in tickers:
        try:
            data = fetch_prices(ticker)
            
            # Insert immutably (or replace if dates overlap)
            for bar in data:
                # Tiingo date format: 2025-01-02T00:00:00.000Z
                date_str = bar["date"][:10]
                
                # We use adjOpen/adjClose to account for splits and dividends
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prices 
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        date_str,
                        bar["adjOpen"],
                        bar["adjHigh"],
                        bar["adjLow"],
                        bar["adjClose"],
                        bar["adjVolume"]
                    )
                )
            conn.commit()
            time.sleep(0.5) # rate limit safety
            
        except Exception as e:
            log.warning("Failed to sync prices for %s: %s", ticker, e)
            
if __name__ == "__main__":
    conn = db.init_db()
    sync_prices(conn)
