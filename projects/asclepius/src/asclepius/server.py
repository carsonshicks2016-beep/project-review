"""MCP server exposing WHOOP data and the correlation engine."""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from . import analyze, store, sync as sync_mod
from .auth import AuthError
from .config import ConfigError
from .frame import METRICS, build_frame, describe_metrics

mcp = MCPServer(
    "asclepius",
    instructions=(
        "Personal WHOOP data with a statistics layer. Call whoop_metrics first "
        "to see what is available, whoop_rank_drivers to find candidate "
        "explanations for an outcome, then whoop_regress to check whether a "
        "candidate survives once confounders are held fixed. Correlations here "
        "are associations, not causes."
    ),
)

_conn = None
_frame_cache: tuple[int, pd.DataFrame] | None = None


def _db():
    global _conn
    if _conn is None:
        _conn = store.connect()
    return _conn


def _frame(force: bool = False) -> pd.DataFrame:
    """Cached daily frame, invalidated whenever the row count changes."""
    global _frame_cache
    conn = _db()
    version = sum(store.counts(conn).values())
    if force or _frame_cache is None or _frame_cache[0] != version:
        _frame_cache = (version, build_frame(conn))
    return _frame_cache[1]


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Infinity with None throughout a result payload.

    Both are legal Python floats and json.dumps will happily emit them, but
    `NaN` and `Infinity` are not valid JSON and strict parsers on the other end
    of the MCP transport reject the whole message. An undefined statistic is
    better delivered as null than as a payload that fails to decode.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _guard(fn, *a, **kw) -> Any:
    """Turn expected failures into readable results instead of tracebacks."""
    try:
        return _json_safe(fn(*a, **kw))
    except (analyze.NotEnoughData, AuthError, ConfigError) as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _require_data() -> dict | None:
    df = _frame()
    if df.empty:
        return {
            "error": "Local cache is empty. Run the whoop_sync tool first "
            "(or `uv run python -m asclepius.cli sync --full` in a terminal)."
        }
    return None


# ---------------------------------------------------------------- data tools


@mcp.tool()
def whoop_status() -> dict:
    """Show what WHOOP data is cached locally: record counts and date coverage."""

    def run():
        conn = _db()
        c = store.counts(conn)
        lo, hi = store.date_range(conn)
        df = _frame()
        return {
            "records": c,
            "date_range": {"first": lo, "last": hi},
            "days_in_frame": int(len(df)),
            "last_sync_utc": store.get_meta(conn, sync_mod.WATERMARK),
            "database": str(store.DB_FILE),
        }

    return _guard(run)


@mcp.tool()
def whoop_sync(full: bool = False, lookback_days: int = 1095) -> dict:
    """Pull new WHOOP records into the local cache.

    Args:
        full: Re-download everything in the lookback window instead of resuming
            from the last sync watermark.
        lookback_days: How far back to reach on a full backfill.
    """

    def run():
        res = sync_mod.sync(_db(), full=full, lookback_days=lookback_days)
        _frame(force=True)
        conn = _db()
        lo, hi = store.date_range(conn)
        return {
            "synced": res,
            "totals": store.counts(conn),
            "date_range": {"first": lo, "last": hi},
        }

    return _guard(run)


@mcp.tool()
def whoop_metrics() -> dict:
    """List every available metric with its unit, coverage, and summary stats.

    Call this first to see what can be fed to the analysis tools.
    """

    def run():
        if (err := _require_data()):
            return err
        return {"metrics": describe_metrics(_frame())}

    return _guard(run)


@mcp.tool()
def whoop_daily(
    start: str | None = None,
    end: str | None = None,
    metrics: list[str] | None = None,
    limit: int = 120,
) -> dict:
    """Return day-by-day values.

    Args:
        start: ISO date, inclusive.
        end: ISO date, inclusive.
        metrics: Which columns to return. Defaults to a compact core set.
        limit: Cap on rows returned, most recent first.
    """

    def run():
        if (err := _require_data()):
            return err
        df = _frame()
        cols = metrics or [
            "recovery_score",
            "hrv_rmssd_milli",
            "resting_hr",
            "sleep_hours",
            "sleep_performance",
            "strain",
            "workout_count",
        ]
        missing = [c for c in cols if c not in df.columns]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return {"error": f"None of those metrics exist. Unknown: {missing}"}

        sub = analyze._slice(df, start, end)[cols].tail(limit)
        recs = [
            {"date": d.date().isoformat(),
             **{c: (None if pd.isna(v) else round(float(v), 3)) for c, v in row.items()}}
            for d, row in sub.iterrows()
        ]
        return {"days": len(recs), "unknown_metrics": missing, "data": recs}

    return _guard(run)


# ------------------------------------------------------------ analysis tools


@mcp.tool()
def whoop_correlate(
    driver: str,
    outcome: str,
    max_lag: int = 3,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Correlate one metric against another across time lags.

    Lag 1 means the driver's value today is compared against the outcome
    tomorrow. p-values are corrected for autocorrelation and for the number of
    lags tested, so they are stricter than a plain correlation.

    Args:
        driver: The metric hypothesized to lead, e.g. "strain".
        outcome: The metric hypothesized to follow, e.g. "recovery_score".
        max_lag: Highest lag in days to test.
        start: ISO date, inclusive.
        end: ISO date, inclusive.
    """

    def run():
        if (err := _require_data()):
            return err
        return analyze.lagged_correlation(
            _frame(), driver, outcome, max_lag=max_lag, start=start, end=end
        )

    return _guard(run)


@mcp.tool()
def whoop_rank_drivers(
    outcome: str = "recovery_score",
    lag: int = 1,
    start: str | None = None,
    end: str | None = None,
    top_n: int = 15,
) -> dict:
    """Screen every metric against one outcome and rank by association strength.

    Use this to find candidates, then confirm them with whoop_regress. Results
    carry FDR q-values because screening this many metrics at once produces
    false positives at raw p<0.05.

    Args:
        outcome: Metric to explain, e.g. "recovery_score".
        lag: Days the driver leads the outcome. 1 is the usual choice for
            "what I did today shows up tomorrow".
        top_n: How many ranked results to return.
    """

    def run():
        if (err := _require_data()):
            return err
        return analyze.rank_drivers(
            _frame(), outcome, lag=lag, start=start, end=end, top_n=top_n
        )

    return _guard(run)


@mcp.tool()
def whoop_regress(
    outcome: str,
    drivers: list[str],
    controls: list[str] | None = None,
    lag: int = 0,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Fit a regression to isolate each driver's effect with covariates held fixed.

    This is how to tell a real effect from a proxy. If late bedtimes correlate
    with bad recovery, put bedtime_hr in drivers and sleep_hours in controls: a
    surviving bedtime coefficient means timing matters beyond mere duration.

    Args:
        outcome: Metric to explain.
        drivers: Metrics of interest.
        controls: Confounders to hold fixed.
        lag: Days the predictors lead the outcome.
    """

    def run():
        if (err := _require_data()):
            return err
        return analyze.regress(
            _frame(), outcome, drivers, controls or [], lag=lag, start=start, end=end
        )

    return _guard(run)


@mcp.tool()
def whoop_compare_periods(
    a_start: str,
    a_end: str,
    b_start: str,
    b_end: str,
    metrics: list[str] | None = None,
) -> dict:
    """Compare metric averages between two date windows.

    Good for before/after questions: a training block, a travel stretch, a
    change in habits. Reports the difference, effect size, and significance.
    """

    def run():
        if (err := _require_data()):
            return err
        cols = metrics or [
            "recovery_score",
            "hrv_rmssd_milli",
            "resting_hr",
            "sleep_hours",
            "sleep_performance",
            "strain",
            "respiratory_rate",
        ]
        return analyze.compare_periods(_frame(), cols, a_start, a_end, b_start, b_end)

    return _guard(run)


@mcp.tool()
def whoop_anomalies(
    last_n_days: int = 14,
    baseline_days: int = 30,
    z_threshold: float = 2.0,
) -> dict:
    """Flag recent days where vitals departed from their trailing baseline.

    Days with two or more metrics deviating together are the meaningful ones —
    that combination often precedes symptoms of illness. Descriptive only, not
    a diagnosis.
    """

    def run():
        if (err := _require_data()):
            return err
        return analyze.anomalies(
            _frame(),
            baseline_days=baseline_days,
            z_threshold=z_threshold,
            last_n_days=last_n_days,
        )

    return _guard(run)


def main() -> int:
    try:
        mcp.run()
    except KeyboardInterrupt:
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"asclepius failed to start: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
