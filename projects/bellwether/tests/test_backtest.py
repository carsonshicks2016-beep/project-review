import pytest
from bellwether import backtest
import sqlite3

def test_calculate_slippage():
    # Synthetic fill bar
    # Open: 10, High: 11, Low: 9, Close: 10.5
    fill_bar = {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5}
    
    # Mid = 10.0
    # Range = 2 / 10 = 0.2 (20% daily range)
    # Half spread approx = 10% of 20% = 2%
    
    # Impact: adv = 1,000,000. size = 50,000
    # sqrt(50,000 / 1,000,000) = sqrt(0.05) = ~0.223
    # impact = 0.1 * 0.2 * 0.223 = ~0.00446 (0.44%)
    
    # adv < 2M -> multiplier = 5
    # (2% + 0.44%) * 5 = 12.2% slippage penalty!
    
    slippage = backtest.calculate_slippage(fill_bar, 1_000_000)
    assert 0.10 < slippage < 0.15 # should be around 12.2%
    
    # High liquidity (ADV 20M) -> multiplier = 1
    # sqrt(50k / 20M) = sqrt(0.0025) = 0.05
    # impact = 0.1 * 0.2 * 0.05 = 0.001 (0.1%)
    # total = (2% + 0.1%) * 1 = 2.1%
    slippage_liquid = backtest.calculate_slippage(fill_bar, 20_000_000)
    assert 0.02 < slippage_liquid < 0.03

def test_point_in_time_enforcement():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE prices (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    
    # Insert dummy data
    conn.execute("INSERT INTO prices VALUES ('XYZ', '2026-06-20', 10, 11, 9, 10, 1000)")
    conn.execute("INSERT INTO prices VALUES ('XYZ', '2026-06-21', 11, 12, 10, 11, 1000)")
    conn.execute("INSERT INTO prices VALUES ('XYZ', '2026-06-22', 12, 13, 11, 12, 1000)")
    
    # Signal arrived exactly AT open on 6-20. We still must NOT use 6-20 data. 
    # We use 6-21.
    bar = backtest.get_next_trading_open(conn, "XYZ", "2026-06-20T09:30:00.000Z")
    assert bar["date"] == "2026-06-21"
    assert bar["open"] == 11.0
    
    # Future price (offset 1 day from fill date 6-21) -> should be 6-22
    fwd = backtest.get_future_price(conn, "XYZ", "2026-06-21", 1)
    assert fwd["date"] == "2026-06-22"
    assert fwd["close"] == 12.0
