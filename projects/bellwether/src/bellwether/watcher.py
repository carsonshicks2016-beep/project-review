"""The filing watcher (Phase 2 / M1) — pure plumbing, no AI.

Hears the market for the pond: re-polls each in-universe company's EDGAR
submissions feed, stores new filings (deduped on accession_no), downloads the
primary document, and logs capture latency vs the filing's acceptance time.

Mechanism note: for a small pond, polling each company's submissions.json is
simpler and friendlier than filtering EDGAR's global "current" firehose, and
gives the same near-real-time result.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

from . import db, sec
from .config import ROOT
from .logging_setup import get_logger

log = get_logger("bellwether.watcher")

FILINGS_DIR = ROOT / "data" / "filings"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def universe_ciks(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return [(cik, ticker)] for in-universe companies."""
    rows = conn.execute(
        "SELECT cik, ticker FROM companies WHERE in_universe=1 AND cik IS NOT NULL"
    ).fetchall()
    return [(r["cik"], r["ticker"]) for r in rows]


def store_filing(conn: sqlite3.Connection, f: dict) -> bool:
    """Insert one filing; return True if it was new (False if already seen)."""
    cur = conn.execute(
        """INSERT OR IGNORE INTO filings
           (cik, form_type, accession_no, filed_date, acceptance_datetime,
            url, fetched_at)
           VALUES (?,?,?,?,?,?,?)""",
        (f["cik"], f["form_type"], f["accession_no"], f["filed_date"],
         f["acceptance_datetime"], f["url"], _now().isoformat()),
    )
    return cur.rowcount > 0


def _latency_minutes(acceptance: str) -> float | None:
    if not acceptance:
        return None
    try:
        t = dt.datetime.fromisoformat(acceptance.replace("Z", "+00:00"))
        return (_now() - t).total_seconds() / 60.0
    except ValueError:
        return None


def poll_company(conn: sqlite3.Connection, cik: str, ticker: str,
                 limit: int | None = None, download: bool = True) -> int:
    """Poll one company; store new filings. Returns count of new filings."""
    try:
        subs = sec.fetch_submissions(cik)
    except Exception as e:
        log.warning("%s (%s): submissions fetch failed: %s", ticker, cik, e)
        return 0
    filings = sec.parse_recent(subs, cik)
    if limit is not None:
        filings = filings[:limit]
    new = 0
    for f in filings:
        if store_filing(conn, f):
            new += 1
            lat = _latency_minutes(f["acceptance_datetime"])
            lat_s = f"{lat:.0f}m" if lat is not None else "?"
            log.info("NEW %-5s %-8s %s  (captured +%s)",
                     ticker, f["form_type"], f["accession_no"], lat_s)
            if download:
                download_doc(conn, f)
    conn.commit()
    return new


def download_doc(conn: sqlite3.Connection, f: dict) -> None:
    """Download a filing's primary document to data/filings/, record raw_path.

    Form 4 / 13D primary docs are .xml; keep whatever extension EDGAR uses.
    """
    if not f["url"]:
        return
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    ext = f["url"].rsplit(".", 1)[-1].lower()
    ext = ext if ext in ("htm", "html", "xml", "txt") else "html"
    path = FILINGS_DIR / f"{f['accession_no']}.{ext}"
    try:
        resp = sec.edgar_get(f["url"])
        path.write_bytes(resp.content)
        conn.execute(
            "UPDATE filings SET raw_path=? WHERE accession_no=?",
            (str(path.relative_to(ROOT)), f["accession_no"]),
        )
    except Exception as e:
        log.warning("doc download failed for %s: %s", f["accession_no"], e)


def backfill(conn: sqlite3.Connection, limit: int | None = 40,
             download: bool = False) -> int:
    """Seed recent filing history across the whole pond. Returns total new."""
    total = 0
    for cik, ticker in universe_ciks(conn):
        n = poll_company(conn, cik, ticker, limit=limit, download=download)
        total += n
        log.info("backfill %-5s: +%d", ticker, n)
    log.info("backfill complete: %d new filings", total)
    return total


POLL_DEPTH = 25  # real-time: new filings appear at the top; no need to re-scan history


def poll_once(conn: sqlite3.Connection, download: bool = True) -> int:
    """One real-time sweep of the pond for new filings. Returns total new.

    Only inspects the most-recent POLL_DEPTH filings per company — anything new
    is at the top of the feed, so re-scanning the full history every cycle is
    wasted requests.
    """
    total = sum(
        poll_company(conn, cik, ticker, limit=POLL_DEPTH, download=download)
        for cik, ticker in universe_ciks(conn)
    )
    log.info("poll complete: %d new filings", total)
    return total
