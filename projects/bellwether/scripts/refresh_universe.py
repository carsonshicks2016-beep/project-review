"""Step 1.8 — Universe auto-refresh.

Re-runs metrics pulling (1.4-1.5) and screening (1.6) for the existing companies
in the database, updating them idempotently.
"""

from __future__ import annotations

import datetime as dt
import time

from bellwether import db, marketdata, screen
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.refresh_universe")


def main() -> None:
    conn = db.init_db()
    
    # Get all companies currently in the database
    rows = conn.execute("SELECT cik, ticker, in_universe FROM companies WHERE cik IS NOT NULL").fetchall()
    log.info("Refreshing metrics for %d companies...", len(rows))
    
    for row in rows:
        cik = row["cik"]
        t = row["ticker"]
        old_in_univ = bool(row["in_universe"])
        
        try:
            f = marketdata.universe_fields(t)
            # Fetch institutional ownership from yfinance if available
            import yfinance as yf
            info = yf.Ticker(t).info
            inst_own = info.get("heldPercentInstitutions")
        except Exception as e:
            log.warning("%s: yfinance error %s", t, e)
            f = {"analyst_count": None, "mktcap": None, "price": None, "adv_usd": None}
            inst_own = None
            
        in_univ, fails = screen.screen(f)
        
        conn.execute(
            """UPDATE companies SET
                mktcap = ?,
                adv_usd = ?,
                analyst_count = ?,
                inst_own_pct = ?,
                in_universe = ?
               WHERE cik = ?""",
            (f["mktcap"], f["adv_usd"], f["analyst_count"], inst_own, int(in_univ), cik)
        )
        
        if old_in_univ != in_univ:
            direction = "ADDED to" if in_univ else "REMOVED from"
            log.info("%-6s %s universe (fails: %s)", t, direction, ",".join(fails) if fails else "none")
            
        time.sleep(0.3)  # be gentle to yfinance
        
    conn.commit()
    log.info("Refresh complete.")
    
if __name__ == "__main__":
    main()
