"""SQLite connection + schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with foreign keys enabled and row access by name."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create all tables from schema.sql (idempotent). Returns the connection."""
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    # Dynamic migration to add processed_at column if it does not exist
    try:
        conn.execute("ALTER TABLE filings ADD COLUMN processed_at TEXT")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    return conn


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]

def get_unprocessed_filings(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    """Get filings that have a downloaded raw_path but have not been processed yet."""
    return conn.execute(
        """
        SELECT f.*, c.ticker 
        FROM filings f
        JOIN companies c ON f.cik = c.cik
        WHERE f.raw_path IS NOT NULL 
          AND f.processed_at IS NULL
        ORDER BY f.acceptance_datetime ASC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

def mark_filing_processed(conn: sqlite3.Connection, filing_id: int) -> None:
    """Mark a filing as processed by setting processed_at."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "UPDATE filings SET processed_at = ? WHERE id = ?",
        (now, filing_id)
    )
    conn.commit()

def insert_signal(conn: sqlite3.Connection, filing_id: int, signal, model_name: str) -> int:
    """Immutably insert an extracted signal. Returns the new signal ID."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO signals 
        (filing_id, catalyst_type, direction, materiality, confidence, summary, evidence_quote, model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filing_id,
            signal.catalyst_type,
            signal.direction,
            signal.materiality,
            signal.confidence,
            signal.summary,
            signal.evidence_quote,
            model_name,
            now
        )
    )
    conn.commit()
    return cur.lastrowid
