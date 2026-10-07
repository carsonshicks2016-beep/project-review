"""Step 1.0 — Universe sanity check (the GATE before loading any pond).

The whole universe rests on `analyst_count`. The FMP free tier paywalls this for
micro-caps (402), so our source is yfinance (Yahoo's numberOfAnalystOpinions).
Before trusting it, compare yfinance's count against a small set of names where
Hudson/Dad already know the true count.

Input:  data/known_analyst_counts.csv  with columns:  ticker,known_count
Output: a per-name comparison + the overall discrepancy rate.

GATE: if >30% of names with a true count <= 3 disagree with FMP, we build the
universe MANUALLY (Hudson's lane) until FMP earns trust. Don't load a rotten pond.
"""

from __future__ import annotations

import csv
from pathlib import Path

from bellwether import marketdata
from bellwether.config import ROOT
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.sanity")

INPUT = ROOT / "data" / "known_analyst_counts.csv"
DISCREPANCY_GATE = 0.30  # >30% mismatch among low-coverage names => fail


def load_known() -> list[tuple[str, int]]:
    if not INPUT.exists():
        raise SystemExit(
            f"Missing {INPUT}. Create it with columns: ticker,known_count\n"
            "Ask Hudson/Dad for ~10 names where they KNOW the analyst count."
        )
    out = []
    with INPUT.open() as f:
        for row in csv.DictReader(f):
            out.append((row["ticker"].strip().upper(), int(row["known_count"])))
    return out


def main() -> None:
    known = load_known()
    log.info("checking %d names against yfinance...", len(known))

    rows = []
    for ticker, truth in known:
        try:
            got = marketdata.analyst_count(ticker)
        except Exception as e:  # network / symbol issues
            got = None
            log.warning("%s: yfinance error %s", ticker, e)
        # "agreement" is loose: within 1 analyst counts as a match.
        match = got is not None and abs(got - truth) <= 1
        rows.append((ticker, truth, got, match))

    print(f"\n{'ticker':<8}{'known':>6}{'yf':>6}  match")
    print("-" * 32)
    for ticker, truth, got, match in rows:
        disp = "—" if got is None else str(got)
        print(f"{ticker:<8}{truth:>6}{disp:>6}  {'OK' if match else 'MISS'}")

    # The gate only judges the low-coverage names — that's where we actually live.
    low = [r for r in rows if r[1] <= 3]
    if not low:
        log.warning("no names with known count <=3 — add some; that's our universe")
        return
    misses = sum(1 for r in low if not r[3])
    rate = misses / len(low)
    print(f"\nlow-coverage (<=3) names: {len(low)}, misses: {misses}, "
          f"discrepancy: {rate:.0%}")

    if rate > DISCREPANCY_GATE:
        print(f"\n❌ GATE FAILED ({rate:.0%} > {DISCREPANCY_GATE:.0%}). "
              "Build the universe MANUALLY until FMP proves reliable.")
    else:
        print(f"\n✅ GATE PASSED ({rate:.0%} <= {DISCREPANCY_GATE:.0%}). "
              "FMP analyst counts are trustworthy enough to load the pond.")


if __name__ == "__main__":
    main()
