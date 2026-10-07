"""SQLite cache for WHOOP records.

Scores are flattened into columns because that is what the analysis layer wants,
but every row also keeps its raw JSON so adding a field later is a migration
rather than a full re-backfill.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DB_FILE, ensure_home

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id              INTEGER PRIMARY KEY,
    start_at        TEXT NOT NULL,
    end_at          TEXT,
    tz_offset       TEXT,
    local_date      TEXT,
    score_state     TEXT,
    strain          REAL,
    kilojoule       REAL,
    avg_hr          INTEGER,
    max_hr          INTEGER,
    updated_at      TEXT,
    raw             TEXT
);
CREATE INDEX IF NOT EXISTS idx_cycles_date ON cycles(local_date);

CREATE TABLE IF NOT EXISTS recovery (
    cycle_id          INTEGER PRIMARY KEY,
    sleep_id          TEXT,
    score_state       TEXT,
    user_calibrating  INTEGER,
    recovery_score    REAL,
    resting_hr        REAL,
    hrv_rmssd_milli   REAL,
    spo2_percentage   REAL,
    skin_temp_celsius REAL,
    updated_at        TEXT,
    raw               TEXT
);

CREATE TABLE IF NOT EXISTS sleeps (
    id                TEXT PRIMARY KEY,
    cycle_id          INTEGER,
    start_at          TEXT NOT NULL,
    end_at            TEXT,
    tz_offset         TEXT,
    local_date        TEXT,
    nap               INTEGER,
    score_state       TEXT,
    in_bed_milli      INTEGER,
    awake_milli       INTEGER,
    no_data_milli     INTEGER,
    light_milli       INTEGER,
    sws_milli         INTEGER,
    rem_milli         INTEGER,
    sleep_cycle_count INTEGER,
    disturbance_count INTEGER,
    baseline_milli    INTEGER,
    debt_milli        INTEGER,
    strain_need_milli INTEGER,
    nap_need_milli    INTEGER,
    respiratory_rate  REAL,
    performance_pct   REAL,
    consistency_pct   REAL,
    efficiency_pct    REAL,
    updated_at        TEXT,
    raw               TEXT
);
CREATE INDEX IF NOT EXISTS idx_sleeps_date ON sleeps(local_date);
CREATE INDEX IF NOT EXISTS idx_sleeps_cycle ON sleeps(cycle_id);

CREATE TABLE IF NOT EXISTS workouts (
    id                  TEXT PRIMARY KEY,
    start_at            TEXT NOT NULL,
    end_at              TEXT,
    tz_offset           TEXT,
    local_date          TEXT,
    sport_name          TEXT,
    sport_id            INTEGER,
    score_state         TEXT,
    strain              REAL,
    avg_hr              INTEGER,
    max_hr              INTEGER,
    kilojoule           REAL,
    percent_recorded    REAL,
    distance_meter      REAL,
    altitude_gain_meter REAL,
    zone_zero_milli     INTEGER,
    zone_one_milli      INTEGER,
    zone_two_milli      INTEGER,
    zone_three_milli    INTEGER,
    zone_four_milli     INTEGER,
    zone_five_milli     INTEGER,
    updated_at          TEXT,
    raw                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(local_date);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    ensure_home()
    conn = sqlite3.connect(path or DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


# ------------------------------------------------------------------ helpers


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def local_date(start: str | None, tz_offset: str | None) -> str | None:
    """The calendar date a record belongs to in the wearer's own timezone.

    WHOOP timestamps are UTC and the offset comes alongside as "-05:00", so a
    late-evening workout would otherwise land on the following day.
    """
    dt = _parse_dt(start)
    if dt is None:
        return None
    if tz_offset:
        try:
            sign = -1 if tz_offset[0] == "-" else 1
            hh, mm = tz_offset[1:].split(":")
            dt = dt.astimezone(timezone.utc) + sign * timedelta(
                hours=int(hh), minutes=int(mm)
            )
            return dt.date().isoformat()
        except (ValueError, IndexError):
            pass
    return dt.date().isoformat()


def _g(d: dict | None, *keys, default=None):
    """Nested get that tolerates missing score blocks on unscored records."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _upsert(conn, table: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' for _ in cols)}) "
        f"ON CONFLICT({cols[0]}) DO UPDATE SET "
        + ",".join(f"{c}=excluded.{c}" for c in cols[1:])
    )
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])
    return len(rows)


# ------------------------------------------------------------------ mappers


def map_cycle(r: dict) -> dict:
    return {
        "id": r["id"],
        "start_at": r.get("start"),
        "end_at": r.get("end"),
        "tz_offset": r.get("timezone_offset"),
        "local_date": local_date(r.get("start"), r.get("timezone_offset")),
        "score_state": r.get("score_state"),
        "strain": _g(r, "score", "strain"),
        "kilojoule": _g(r, "score", "kilojoule"),
        "avg_hr": _g(r, "score", "average_heart_rate"),
        "max_hr": _g(r, "score", "max_heart_rate"),
        "updated_at": r.get("updated_at"),
        "raw": json.dumps(r),
    }


def map_recovery(r: dict) -> dict:
    return {
        "cycle_id": r["cycle_id"],
        "sleep_id": r.get("sleep_id"),
        "score_state": r.get("score_state"),
        "user_calibrating": int(bool(_g(r, "score", "user_calibrating", default=False))),
        "recovery_score": _g(r, "score", "recovery_score"),
        "resting_hr": _g(r, "score", "resting_heart_rate"),
        "hrv_rmssd_milli": _g(r, "score", "hrv_rmssd_milli"),
        "spo2_percentage": _g(r, "score", "spo2_percentage"),
        "skin_temp_celsius": _g(r, "score", "skin_temp_celsius"),
        "updated_at": r.get("updated_at"),
        "raw": json.dumps(r),
    }


def map_sleep(r: dict) -> dict:
    ss = _g(r, "score", "stage_summary", default={}) or {}
    sn = _g(r, "score", "sleep_needed", default={}) or {}
    return {
        "id": r["id"],
        "cycle_id": r.get("cycle_id"),
        "start_at": r.get("start"),
        "end_at": r.get("end"),
        "tz_offset": r.get("timezone_offset"),
        "local_date": local_date(r.get("end") or r.get("start"), r.get("timezone_offset")),
        "nap": int(bool(r.get("nap"))),
        "score_state": r.get("score_state"),
        "in_bed_milli": ss.get("total_in_bed_time_milli"),
        "awake_milli": ss.get("total_awake_time_milli"),
        "no_data_milli": ss.get("total_no_data_time_milli"),
        "light_milli": ss.get("total_light_sleep_time_milli"),
        "sws_milli": ss.get("total_slow_wave_sleep_time_milli"),
        "rem_milli": ss.get("total_rem_sleep_time_milli"),
        "sleep_cycle_count": ss.get("sleep_cycle_count"),
        "disturbance_count": ss.get("disturbance_count"),
        "baseline_milli": sn.get("baseline_milli"),
        "debt_milli": sn.get("need_from_sleep_debt_milli"),
        "strain_need_milli": sn.get("need_from_recent_strain_milli"),
        "nap_need_milli": sn.get("need_from_recent_nap_milli"),
        "respiratory_rate": _g(r, "score", "respiratory_rate"),
        "performance_pct": _g(r, "score", "sleep_performance_percentage"),
        "consistency_pct": _g(r, "score", "sleep_consistency_percentage"),
        "efficiency_pct": _g(r, "score", "sleep_efficiency_percentage"),
        "updated_at": r.get("updated_at"),
        "raw": json.dumps(r),
    }


def map_workout(r: dict) -> dict:
    z = _g(r, "score", "zone_durations", default={}) or {}
    return {
        "id": r["id"],
        "start_at": r.get("start"),
        "end_at": r.get("end"),
        "tz_offset": r.get("timezone_offset"),
        "local_date": local_date(r.get("start"), r.get("timezone_offset")),
        "sport_name": r.get("sport_name"),
        "sport_id": r.get("sport_id"),
        "score_state": r.get("score_state"),
        "strain": _g(r, "score", "strain"),
        "avg_hr": _g(r, "score", "average_heart_rate"),
        "max_hr": _g(r, "score", "max_heart_rate"),
        "kilojoule": _g(r, "score", "kilojoule"),
        "percent_recorded": _g(r, "score", "percent_recorded"),
        "distance_meter": _g(r, "score", "distance_meter"),
        "altitude_gain_meter": _g(r, "score", "altitude_gain_meter"),
        "zone_zero_milli": z.get("zone_zero_milli"),
        "zone_one_milli": z.get("zone_one_milli"),
        "zone_two_milli": z.get("zone_two_milli"),
        "zone_three_milli": z.get("zone_three_milli"),
        "zone_four_milli": z.get("zone_four_milli"),
        "zone_five_milli": z.get("zone_five_milli"),
        "updated_at": r.get("updated_at"),
        "raw": json.dumps(r),
    }


MAPPERS = {
    "cycles": map_cycle,
    "recovery": map_recovery,
    "sleeps": map_sleep,
    "workouts": map_workout,
}


def save(conn, table: str, records: Iterable[dict]) -> int:
    mapper = MAPPERS[table]
    rows = [mapper(r) for r in records]
    n = _upsert(conn, table, rows)
    conn.commit()
    return n


def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key,value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def counts(conn) -> dict[str, int]:
    out = {}
    for t in ("cycles", "recovery", "sleeps", "workouts"):
        out[t] = conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
    return out


def date_range(conn) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT MIN(local_date) a, MAX(local_date) b FROM cycles"
    ).fetchone()
    return (row["a"], row["b"])
