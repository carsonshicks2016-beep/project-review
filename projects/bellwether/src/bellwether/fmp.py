"""Financial Modeling Prep client (the `stable` API — legacy v3/v4 are closed
to accounts created after Aug 2025).

We use FMP for the Layer-1 (price/cap/liquidity) and Layer-2 (analyst coverage)
inputs to the universe screen. Analyst count is approximated by summing the
grades-consensus rating buckets — its reliability for micro-caps is exactly what
Step 1.0 (the universe sanity check) is built to verify before we trust it.
"""

from __future__ import annotations

import requests

from . import config

BASE = "https://financialmodelingprep.com/stable"
TIMEOUT = 20


def _get(path: str, **params) -> list | dict:
    params["apikey"] = config.require("FMP_API_KEY")
    r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def profile(symbol: str) -> dict | None:
    """Company profile: price, marketCap, averageVolume, etc. None if missing."""
    data = _get("profile", symbol=symbol)
    return data[0] if data else None


def analyst_count(symbol: str) -> int | None:
    """Approximate # of covering analysts = sum of grades-consensus buckets.

    Returns None if FMP has no consensus row for the symbol (common for the
    truly neglected names we care about — itself a useful signal).
    """
    data = _get("grades-consensus", symbol=symbol)
    if not data:
        return None
    row = data[0]
    return sum(
        int(row.get(k, 0) or 0)
        for k in ("strongBuy", "buy", "hold", "sell", "strongSell")
    )


def adv_usd(symbol: str) -> float | None:
    """Average daily dollar volume = averageVolume * price."""
    p = profile(symbol)
    if not p:
        return None
    vol = p.get("averageVolume")
    price = p.get("price")
    if vol is None or price is None:
        return None
    return float(vol) * float(price)
