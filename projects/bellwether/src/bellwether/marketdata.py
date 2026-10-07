"""Universe market data via yfinance (free, covers micro-caps).

Why not FMP: the FMP free tier paywalls analyst coverage (grades-consensus → 402)
for exactly the small/mid-caps that make up our universe. yfinance exposes
Yahoo's `numberOfAnalystOpinions`, market cap, price, and average volume for
those names at no cost. FMP's free `profile` still works if we want a cross-check
(see fmp.py), but yfinance is the primary universe source.

Caveat: yfinance is an unofficial Yahoo scraper — fine for a personal v1 tool,
but it can break if Yahoo changes things. Step 1.0 validates the analyst counts
against human-known values before we trust them.
"""

from __future__ import annotations

import yfinance as yf


def universe_fields(symbol: str) -> dict:
    """Return the Layer-1/Layer-2 inputs for one ticker.

    Keys: analyst_count, mktcap, price, adv_usd (avg daily $ volume). Any field
    may be None if Yahoo lacks it.
    """
    info = yf.Ticker(symbol).info
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    avg_vol = info.get("averageVolume")
    adv_usd = float(price) * float(avg_vol) if price and avg_vol else None
    return {
        "analyst_count": info.get("numberOfAnalystOpinions"),
        "mktcap": info.get("marketCap"),
        "price": price,
        "adv_usd": adv_usd,
    }


def analyst_count(symbol: str) -> int | None:
    n = yf.Ticker(symbol).info.get("numberOfAnalystOpinions")
    return int(n) if n is not None else None
