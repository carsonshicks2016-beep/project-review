"""Incremental sync from the WHOOP API into the local SQLite cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import store
from .client import WhoopClient

# Records mutate after creation: a cycle is PENDING_SCORE until the day closes,
# and recovery lands hours after the sleep it belongs to. Re-pulling a trailing
# window on every sync lets those late scores overwrite the provisional rows.
OVERLAP_DAYS = 10

WATERMARK = "last_sync_utc"

STREAMS = {
    "cycles": lambda c, s, e: c.cycles(s, e),
    "recovery": lambda c, s, e: c.recoveries(s, e),
    "sleeps": lambda c, s, e: c.sleeps(s, e),
    "workouts": lambda c, s, e: c.workouts(s, e),
}


def sync(
    conn,
    since: datetime | None = None,
    full: bool = False,
    lookback_days: int = 1095,
) -> dict[str, int]:
    """Pull new/updated records. Returns per-stream record counts.

    With no arguments this resumes from the last watermark. `full=True` forces a
    complete backfill over `lookback_days`.
    """
    now = datetime.now(timezone.utc)

    if full or since is None:
        mark = None if full else store.get_meta(conn, WATERMARK)
        if mark:
            since = datetime.fromisoformat(mark) - timedelta(days=OVERLAP_DAYS)
        else:
            since = now - timedelta(days=lookback_days)

    results: dict[str, int] = {}
    with WhoopClient() as client:
        for table, fetch in STREAMS.items():
            results[table] = store.save(conn, table, fetch(client, since, now))

    store.set_meta(conn, WATERMARK, now.isoformat())
    return results
