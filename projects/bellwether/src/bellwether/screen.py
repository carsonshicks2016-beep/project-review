"""The 3-layer universe screen (PLAN.md §7).

Thresholds are intentionally loose for v1 — we calibrate against the actual
output (Step 1.7), not by theorizing. A name is `in_universe` only if it clears
Layer 1 (investability) AND Layer 2's neglect test. Layer 3 (not-a-trap) needs
fundamentals we don't pull yet, so it's left for manual review.
"""

from __future__ import annotations

# Layer 1 — investability floor
MIN_PRICE = 5.0
MIN_CAP = 300e6
MAX_CAP = 3e9
MIN_ADV_USD = 1.5e6

# Layer 2 — neglect
# Calibrated 2026-06-18 (Step 1.7): ≤3 left only 8 names; loosened to ≤4 for a
# slightly wider pond (~11) while staying genuinely neglected.
MAX_ANALYSTS = 4


def screen(fields: dict) -> tuple[bool, list[str]]:
    """Return (in_universe, reasons_failed). Missing data => a fail reason."""
    fails = []
    price = fields.get("price")
    cap = fields.get("mktcap")
    adv = fields.get("adv_usd")
    analysts = fields.get("analyst_count")

    if price is None or price < MIN_PRICE:
        fails.append(f"price<{MIN_PRICE:g}" if price is not None else "price?")
    if cap is None or not (MIN_CAP <= cap <= MAX_CAP):
        fails.append("cap_out" if cap is not None else "cap?")
    if adv is None or adv < MIN_ADV_USD:
        fails.append("adv_low" if adv is not None else "adv?")
    if analysts is None:
        fails.append("analysts?")
    elif analysts > MAX_ANALYSTS:
        fails.append(f"analysts>{MAX_ANALYSTS}")

    return (len(fails) == 0, fails)
