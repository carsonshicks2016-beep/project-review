"""Phase 1 (steps 1.2-1.6) — build the pond.

Reads data/seed_universe.csv (Hudson's list), pulls live fields from yfinance,
resolves CIKs from EDGAR, applies the 3-layer screen, and loads everything into
the `companies` table. Prints a summary so we can calibrate (Step 1.7).

Idempotent: re-running upserts by CIK.
"""

from __future__ import annotations

import csv
import datetime as dt
import time

from bellwether import db, marketdata, screen, sec
from bellwether.config import ROOT
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.load_universe")
SEED = ROOT / "data" / "seed_universe.csv"


def main() -> None:
    seed = list(csv.DictReader(SEED.open()))
    tickers = [r["ticker"].strip().upper() for r in seed]
    log.info("resolving %d CIKs from EDGAR...", len(tickers))
    ciks = sec.resolve_ciks(tickers)

    conn = db.init_db()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []

    for r in seed:
        t = r["ticker"].strip().upper()
        try:
            f = marketdata.universe_fields(t)
        except Exception as e:
            log.warning("%s: yfinance error %s", t, e)
            f = {"analyst_count": None, "mktcap": None, "price": None, "adv_usd": None}
        in_univ, fails = screen.screen(f)
        cik = ciks.get(t)
        rows.append((t, r, f, cik, in_univ, fails))
        if cik:
            conn.execute(
                """INSERT INTO companies
                   (cik,ticker,name,sector,mktcap,adv_usd,analyst_count,
                    in_universe,added_at,notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(cik) DO UPDATE SET
                    ticker=excluded.ticker, mktcap=excluded.mktcap,
                    adv_usd=excluded.adv_usd, analyst_count=excluded.analyst_count,
                    in_universe=excluded.in_universe""",
                (cik, t, r.get("company"), r.get("sector"), f["mktcap"],
                 f["adv_usd"], f["analyst_count"], int(in_univ), now,
                 f"{r.get('orphan_type','')}: {r.get('why','')}"),
            )
        time.sleep(0.3)  # be gentle to yfinance

    conn.commit()

    # --- report ---
    print(f"\n{'tkr':<6}{'cap$B':>7}{'price':>8}{'ADV$M':>8}{'an':>4}  univ  fails")
    print("-" * 70)
    kept = []
    for t, r, f, cik, in_univ, fails in rows:
        cap = f["mktcap"]; adv = f["adv_usd"]; pr = f["price"]; an = f["analyst_count"]
        capb = f"{cap/1e9:.2f}" if cap else "—"
        prc = f"{pr:.2f}" if pr else "—"
        advm = f"{adv/1e6:.1f}" if adv else "—"
        ans = "—" if an is None else str(an)
        flag = "✓" if in_univ else " "
        nocik = "" if cik else " [noCIK]"
        print(f"{t:<6}{capb:>7}{prc:>8}{advm:>8}{ans:>4}   {flag}   {','.join(fails)}{nocik}")
        if in_univ:
            kept.append(t)

    print(f"\nKEPT {len(kept)}/{len(rows)} in universe: {', '.join(kept)}")
    log.info("loaded %d companies; %d in_universe", len(rows), len(kept))


if __name__ == "__main__":
    main()
