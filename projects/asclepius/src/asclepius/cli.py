"""Terminal entry points for setup and maintenance, outside the MCP server."""

from __future__ import annotations

import argparse
import json
import sys

import webbrowser

from . import analyze, console as ui, dashboard as dash, store, sync as sync_mod
from .auth import AuthError
from .config import ConfigError
from .frame import build_frame, describe_metrics


def _counts(d: dict) -> str:
    return "   ".join(f"{k} {v}" for k, v in d.items())


def _span(lo, hi) -> str:
    return f"{lo} → {hi}" if lo and hi else "—"


def cmd_sync(args) -> int:
    conn = store.connect()
    res = sync_mod.sync(conn, full=args.full, lookback_days=args.lookback_days)
    lo, hi = store.date_range(conn)

    ui.open_panel("sync")
    ui.blank()
    ui.kv("fetched", _counts(res), ui.AMBER_HI)
    ui.kv("totals", _counts(store.counts(conn)))
    ui.kv("range", _span(lo, hi))
    ui.blank()
    ui.close_panel()
    return 0


def cmd_status(args) -> int:
    conn = store.connect()
    lo, hi = store.date_range(conn)
    df = build_frame(conn)
    last = store.get_meta(conn, sync_mod.WATERMARK)

    ui.open_panel("status")
    ui.blank()
    ui.kv("database", str(store.DB_FILE))
    ui.kv("records", _counts(store.counts(conn)))
    ui.kv("range", f"{_span(lo, hi)}   ({len(df)} days in frame)")
    ui.kv("last sync", last or "never", ui.AMBER if last else ui.HOT)
    ui.blank()
    ui.close_panel()

    if not last:
        ui.alert("No data cached yet — run `sync --full` to backfill.")
    return 0


def cmd_metrics(args) -> int:
    df = build_frame(store.connect())
    if df.empty:
        ui.alert("No data cached yet — run `sync --full` first.")
        return 1

    rows = describe_metrics(df)
    ui.open_panel("metrics")
    ui.blank()
    ui.row(
        [("METRIC", 26, "l"), ("N", 6, "r"), ("MEAN", 11, "r"),
         ("SD", 10, "r"), ("UNIT", 10, "l")],
        [ui.AMBER_DIM] * 5,
    )
    ui.divider()
    for m in rows:
        ui.row(
            [(m["metric"], 26, "l"), (m["n"], 6, "r"), (m["mean"], 11, "r"),
             (m["sd"], 10, "r"), (m["unit"], 10, "l")],
            [ui.AMBER_HI, ui.AMBER, ui.AMBER, ui.AMBER, ui.AMBER_DIM],
        )
    ui.blank()
    ui.close_panel()
    ui.note(f"{len(rows)} metrics with coverage. Any of these names can be passed "
            "as --outcome, or to whoop_correlate and whoop_regress.")
    return 0


def cmd_drivers(args) -> int:
    df = build_frame(store.connect())
    out = analyze.rank_drivers(df, args.outcome, lag=args.lag, top_n=args.top_n)

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    ui.open_panel("drivers")
    ui.blank()
    ui.kv("outcome", f"{out['outcome_label']}  (lag {out['lag']})", ui.AMBER_HI)
    ui.kv("screened", f"{out['tested']} metrics")
    ui.blank()
    ui.row(
        [("DRIVER", 26, "l"), ("R", 7, "r"), ("STRENGTH", 8, "l"),
         ("Q(FDR)", 9, "r"), ("N", 6, "r")],
        [ui.AMBER_DIM] * 5,
    )
    ui.divider()
    for r in out["results"]:
        bar, tone = ui.gauge(abs(r["r"]), accent=ui.AMBER if r["r"] >= 0 else ui.HOT)
        q = r["q_fdr"]
        survived = q is not None and q < 0.05
        ui.row(
            [(r["metric"], 26, "l"), (f"{r['r']:+.3f}", 7, "r"), (bar, 8, "l"),
             ("—" if q is None else f"{q:.4f}", 9, "r"), (r["n"], 6, "r")],
            [ui.AMBER_HI if survived else ui.AMBER, tone, tone,
             ui.AMBER_HI if survived else ui.AMBER_DIM, ui.AMBER_DIM],
        )
    ui.blank()
    ui.close_panel()

    kept = out["significant_after_fdr"]
    if kept:
        ui.note("Survives FDR at q<0.05: " + ", ".join(kept))
    else:
        ui.alert("Nothing survives FDR correction at q<0.05.")
    ui.note(out["note"])
    return 0


def cmd_dashboard(args) -> int:
    path = dash.build()
    ui.open_panel("dashboard")
    ui.blank()
    ui.kv("written", str(path), ui.AMBER_HI)
    ui.kv("size", f"{path.stat().st_size / 1024:.0f} KB   self-contained, no network")
    ui.blank()
    ui.close_panel()
    if not args.no_open:
        webbrowser.open(path.as_uri())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="asclepius-cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="pull data from the WHOOP API")
    s.add_argument("--full", action="store_true", help="ignore the watermark")
    s.add_argument("--lookback-days", type=int, default=1095)
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("status", help="show cache coverage")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("metrics", help="list available metrics")
    s.set_defaults(func=cmd_metrics)

    s = sub.add_parser("drivers", help="rank drivers of an outcome")
    s.add_argument("--outcome", default="recovery_score")
    s.add_argument("--lag", type=int, default=1)
    s.add_argument("--top-n", type=int, default=15)
    s.add_argument("--json", action="store_true", help="raw JSON instead of the panel")
    s.set_defaults(func=cmd_drivers)

    s = sub.add_parser("dashboard", help="write and open the local HTML dashboard")
    s.add_argument("--no-open", action="store_true", help="write the file only")
    s.set_defaults(func=cmd_dashboard)

    args = p.parse_args()
    try:
        return args.func(args)
    except (AuthError, ConfigError, analyze.NotEnoughData) as e:
        ui.alert(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
