"""Build the daily feature frame that the analysis layer operates on.

Alignment convention, which everything downstream depends on:

    A row is one WHOOP physiological cycle, keyed by its local calendar date.

    `recovery_*` and `sleep_*` on that row describe the night that ENDED that
    morning. `strain`, `workout_*` and `kilojoule` describe the waking day that
    FOLLOWED it.

So within a single row, sleep precedes strain. To ask "does today's training
cost me tomorrow's recovery", you compare strain(t) against recovery(t+1) —
that is lag 1. `lagged_correlation` takes positive lag to mean the driver leads
the outcome, which matches that reading.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

MS_PER_HOUR = 3_600_000.0
MS_PER_MIN = 60_000.0

# Where the clock-hour scale is allowed to wrap. Sleep boundaries fall in two
# bands — bedtimes from late evening to early morning, wake times from early
# morning to just past noon — which leaves late afternoon empty. Cutting there
# keeps both bands contiguous.
#
# Noon is NOT safe, however tempting: wake times run right up to and past it, so
# a cut at 12:00 puts an 11:58 wake and a 12:05 wake ~24 apart.
WRAP_HOUR = 17.0


def _local_dt(ts: str | None, tz: str | None) -> datetime | None:
    if not ts:
        return None
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    if tz:
        try:
            sign = -1 if tz[0] == "-" else 1
            hh, mm = tz[1:].split(":")
            dt = dt + sign * timedelta(hours=int(hh), minutes=int(mm))
        except (ValueError, IndexError):
            pass
    return dt.replace(tzinfo=None)


def _clock_hours_around_midnight(dt: datetime | None) -> float | None:
    """Hour-of-day as signed hours from midnight, wrapping at `WRAP_HOUR`.

    Bedtimes straddle midnight, so raw hour-of-day jumps from 23.9 to 0.1 for a
    20-minute difference. Moving the discontinuity into the late-afternoon gap
    keeps a night monotonic: 23:00 is -1.0, 00:30 is +0.5, 07:00 is +7.0, and a
    wake time either side of noon stays continuous.

    Note the sign of the comparison. `< WRAP_HOUR` instead of `>=` would put the
    wrap back at midnight *and* spread one person's bedtimes across a 48-hour
    range — worse than not correcting at all, and silently so.
    """
    if dt is None:
        return None
    h = dt.hour + dt.minute / 60 + dt.second / 3600
    return h - 24 if h >= WRAP_HOUR else h


def build_frame(conn) -> pd.DataFrame:
    """Return a date-indexed DataFrame, one row per physiological cycle."""
    cycles = pd.read_sql_query(
        "SELECT id AS cycle_id, local_date, strain, kilojoule, avg_hr, max_hr, "
        "score_state FROM cycles WHERE local_date IS NOT NULL",
        conn,
    )
    if cycles.empty:
        return pd.DataFrame()

    rec = pd.read_sql_query(
        "SELECT cycle_id, sleep_id, recovery_score, resting_hr, hrv_rmssd_milli, "
        "spo2_percentage, skin_temp_celsius, user_calibrating FROM recovery",
        conn,
    )

    sleeps = pd.read_sql_query(
        "SELECT id, cycle_id, local_date, start_at, end_at, tz_offset, nap, "
        "in_bed_milli, awake_milli, light_milli, sws_milli, rem_milli, "
        "sleep_cycle_count, disturbance_count, baseline_milli, debt_milli, "
        "respiratory_rate, performance_pct, consistency_pct, efficiency_pct "
        "FROM sleeps",
        conn,
    )

    workouts = pd.read_sql_query(
        "SELECT local_date, strain, kilojoule, start_at, end_at, sport_name, "
        "zone_three_milli, zone_four_milli, zone_five_milli FROM workouts "
        "WHERE local_date IS NOT NULL",
        conn,
    )

    df = cycles.merge(rec, on="cycle_id", how="left")

    # --- main sleep (naps handled separately) -----------------------------
    # A cycle's sleep is NOT the row sharing its cycle_id. Per the API spec a
    # recovery describes readiness *for* its cycle and carries its own
    # sleep_id, pointing at the night that ended that morning — whereas a
    # sleep's own cycle_id refers to the cycle it occurred during, i.e. the
    # previous one. Joining on cycle_id would pair each day's strain with the
    # night that came after it and invert every lag result, so recovery.sleep_id
    # is the authoritative link, with an end-date match as fallback for cycles
    # that have no scored recovery.
    main = sleeps[sleeps["nap"] == 0].copy()
    if not main.empty:
        for c in ("light_milli", "sws_milli", "rem_milli"):
            main[c] = pd.to_numeric(main[c], errors="coerce")
        main["asleep_milli"] = main[["light_milli", "sws_milli", "rem_milli"]].sum(
            axis=1, min_count=1
        )
        main["_bed_dt"] = [
            _local_dt(s, t) for s, t in zip(main["start_at"], main["tz_offset"])
        ]
        main["_wake_dt"] = [
            _local_dt(e, t) for e, t in zip(main["end_at"], main["tz_offset"])
        ]
        main["bedtime_hr"] = [_clock_hours_around_midnight(d) for d in main["_bed_dt"]]
        main["waketime_hr"] = [_clock_hours_around_midnight(d) for d in main["_wake_dt"]]

        # If a night is split into two records, keep the longest.
        main = main.sort_values("asleep_milli", ascending=False).drop_duplicates(
            "id", keep="first"
        )

        sleep_cols = {
            "sleep_hours": main["asleep_milli"] / MS_PER_HOUR,
            "in_bed_hours": pd.to_numeric(main["in_bed_milli"], errors="coerce")
            / MS_PER_HOUR,
            "awake_hours": pd.to_numeric(main["awake_milli"], errors="coerce")
            / MS_PER_HOUR,
            "rem_hours": main["rem_milli"] / MS_PER_HOUR,
            "sws_hours": main["sws_milli"] / MS_PER_HOUR,
            "light_hours": main["light_milli"] / MS_PER_HOUR,
            "sleep_debt_hours": pd.to_numeric(main["debt_milli"], errors="coerce")
            / MS_PER_HOUR,
            "sleep_performance": main["performance_pct"],
            "sleep_consistency": main["consistency_pct"],
            "sleep_efficiency": main["efficiency_pct"],
            "respiratory_rate": main["respiratory_rate"],
            "sleep_cycles": main["sleep_cycle_count"],
            "disturbances": main["disturbance_count"],
            "bedtime_hr": main["bedtime_hr"],
            "waketime_hr": main["waketime_hr"],
            "sleep_id": main["id"],
            "sleep_end_date": main["local_date"],
        }
        sf = pd.DataFrame(sleep_cols)

        # Resolve each cycle to its sleep: recovery.sleep_id first, then the
        # sleep that ended on that cycle's own calendar date.
        by_date = (
            sf.dropna(subset=["sleep_end_date"])
            .sort_values("sleep_hours", ascending=False)
            .drop_duplicates("sleep_end_date", keep="first")
        )
        date_to_sid = dict(zip(by_date["sleep_end_date"], by_date["sleep_id"]))

        known = set(sf["sleep_id"])
        df["_sid"] = df["sleep_id"].where(
            df["sleep_id"].isin(known), df["local_date"].map(date_to_sid)
        )
        df = df.merge(
            sf.drop(columns=["sleep_end_date"]),
            left_on="_sid",
            right_on="sleep_id",
            how="left",
            suffixes=("_rec", ""),
        ).drop(columns=["_sid"], errors="ignore")

    # --- naps --------------------------------------------------------------
    naps = sleeps[sleeps["nap"] == 1].copy()
    if not naps.empty:
        for c in ("light_milli", "sws_milli", "rem_milli"):
            naps[c] = pd.to_numeric(naps[c], errors="coerce")
        naps["asleep_milli"] = naps[["light_milli", "sws_milli", "rem_milli"]].sum(
            axis=1, min_count=1
        )
        agg = (
            naps.groupby("local_date")
            .agg(
                nap_count=("id", "count"),
                nap_minutes=("asleep_milli", lambda s: s.sum() / MS_PER_MIN),
            )
            .reset_index()
        )
        df = df.merge(agg, on="local_date", how="left")

    # --- workouts ----------------------------------------------------------
    if not workouts.empty:
        w = workouts.copy()
        w["_dur_min"] = [
            (
                (
                    datetime.fromisoformat(e.replace("Z", "+00:00"))
                    - datetime.fromisoformat(s.replace("Z", "+00:00"))
                ).total_seconds()
                / 60
                if s and e
                else np.nan
            )
            for s, e in zip(w["start_at"], w["end_at"])
        ]
        for c in ("zone_three_milli", "zone_four_milli", "zone_five_milli"):
            w[c] = pd.to_numeric(w[c], errors="coerce")
        w["_hi_min"] = w[
            ["zone_three_milli", "zone_four_milli", "zone_five_milli"]
        ].sum(axis=1, min_count=1) / MS_PER_MIN

        agg = (
            w.groupby("local_date")
            .agg(
                workout_count=("strain", "count"),
                workout_strain=("strain", "sum"),
                workout_minutes=("_dur_min", "sum"),
                high_intensity_minutes=("_hi_min", "sum"),
                top_sport=(
                    "sport_name",
                    lambda s: s.mode().iat[0] if not s.mode().empty else None,
                ),
            )
            .reset_index()
        )
        df = df.merge(agg, on="local_date", how="left")

    # --- finalize ----------------------------------------------------------
    df["date"] = pd.to_datetime(df["local_date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")

    # Reindex onto a gap-free calendar. Missing days must exist as NaN rows or
    # every lag operation would silently shift across the gap.
    if len(df) > 1:
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.index.name = "date"

    for c in ("workout_count", "workout_strain", "workout_minutes",
              "high_intensity_minutes", "nap_count", "nap_minutes"):
        if c in df:
            df[c] = df[c].fillna(0.0)

    return df


# Columns worth offering as drivers/outcomes, with human labels and units.
METRICS: dict[str, tuple[str, str]] = {
    "recovery_score": ("Recovery", "%"),
    "hrv_rmssd_milli": ("HRV (RMSSD)", "ms"),
    "resting_hr": ("Resting heart rate", "bpm"),
    "spo2_percentage": ("Blood oxygen", "%"),
    "skin_temp_celsius": ("Skin temperature", "°C"),
    "strain": ("Day strain", "0-21"),
    "kilojoule": ("Energy burned", "kJ"),
    "avg_hr": ("Average heart rate", "bpm"),
    "max_hr": ("Max heart rate", "bpm"),
    "sleep_hours": ("Sleep duration", "h"),
    "in_bed_hours": ("Time in bed", "h"),
    "awake_hours": ("Awake in bed", "h"),
    "rem_hours": ("REM sleep", "h"),
    "sws_hours": ("Deep (SWS) sleep", "h"),
    "light_hours": ("Light sleep", "h"),
    "sleep_debt_hours": ("Sleep debt", "h"),
    "sleep_performance": ("Sleep performance", "%"),
    "sleep_consistency": ("Sleep consistency", "%"),
    "sleep_efficiency": ("Sleep efficiency", "%"),
    "respiratory_rate": ("Respiratory rate", "rpm"),
    "sleep_cycles": ("Sleep cycles", "count"),
    "disturbances": ("Disturbances", "count"),
    "bedtime_hr": ("Bedtime", "h from midnight"),
    "waketime_hr": ("Wake time", "h from midnight"),
    "nap_count": ("Naps", "count"),
    "nap_minutes": ("Nap duration", "min"),
    "workout_count": ("Workouts", "count"),
    "workout_strain": ("Workout strain", "sum"),
    "workout_minutes": ("Workout duration", "min"),
    "high_intensity_minutes": ("Time in HR zones 3-5", "min"),
}


def describe_metrics(df: pd.DataFrame) -> list[dict]:
    out = []
    for col, (label, unit) in METRICS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n = int(s.notna().sum())
        if n == 0:
            continue
        out.append(
            {
                "metric": col,
                "label": label,
                "unit": unit,
                "n": n,
                "mean": round(float(s.mean()), 3),
                "sd": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
            }
        )
    return out
