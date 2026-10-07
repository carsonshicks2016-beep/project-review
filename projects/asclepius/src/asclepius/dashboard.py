"""Build the local console dashboard.

One self-contained HTML file. The whole daily frame is embedded as JSON and
every chart, statistic and interaction is computed in the browser — which is
what makes the thing fully interactive without a server, and what keeps it
consistent with the promise in PRIVACY.md that nothing leaves the machine.

Markup, styling and behaviour live in `web/` as real .html/.css/.js files and
are inlined here at build time, so they stay editable and syntax-highlighted
rather than being buried in Python string literals.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import analyze, store
from .config import HOME
from .frame import METRICS, MS_PER_MIN, build_frame

OUT_FILE = HOME / "dashboard.html"
WEB = Path(__file__).resolve().parent / "web"
LOGO = Path(__file__).resolve().parents[2] / "logo.png"


def _col(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), 4) for v in series]


def _workouts(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT local_date, sport_name, strain, start_at, end_at, kilojoule "
        "FROM workouts WHERE local_date IS NOT NULL ORDER BY local_date"
    ).fetchall()
    out = []
    for r in rows:
        start, end = r["start_at"], r["end_at"]
        minutes = 0.0
        if start and end:
            try:
                a = datetime.fromisoformat(start.replace("Z", "+00:00"))
                b = datetime.fromisoformat(end.replace("Z", "+00:00"))
                minutes = max(0.0, (b - a).total_seconds() / 60.0)
            except ValueError:
                minutes = 0.0
        out.append({
            "date": r["local_date"],
            "sport": r["sport_name"] or "activity",
            "strain": None if r["strain"] is None else round(float(r["strain"]), 2),
            "minutes": round(minutes, 1),
        })
    return out


def payload(df: pd.DataFrame, conn) -> dict:
    """Columnar — far smaller than row objects, and what the charts want."""
    keys = [c for c in METRICS if c in df.columns and df[c].notna().any()]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "metrics": {k: _col(df[k]) for k in keys},
        "labels": {k: METRICS[k][0] for k in keys},
        "units": {k: METRICS[k][1] for k in keys},
        "workouts": _workouts(conn),
        "counts": dict(store.counts(conn)),
    }


def render(df: pd.DataFrame, conn) -> str:
    data = payload(df, conn)
    logo = ""
    if LOGO.exists():
        b64 = base64.b64encode(LOGO.read_bytes()).decode()
        logo = f'<img src="data:image/png;base64,{b64}" alt="">'

    counts = " · ".join(f"{k.upper()} {v}" for k, v in data["counts"].items())
    html = (WEB / "index.html").read_text(encoding="utf-8")

    # Plain replace, not str.format or %-formatting: the CSS and JS are full of
    # braces and percent signs that either of those would try to interpret.
    for token, value in (
        ("__CSS__", (WEB / "app.css").read_text(encoding="utf-8")),
        ("__JS__", (WEB / "app.js").read_text(encoding="utf-8")),
        ("__DATA__", json.dumps(data, separators=(",", ":"))),
        ("__LOGO__", logo),
        ("__COUNTS__", counts),
        ("__FROM__", df.index[0].strftime("%Y-%m-%d")),
        ("__TO__", df.index[-1].strftime("%Y-%m-%d")),
        ("__STAMP__", datetime.now().strftime("%d %b %Y %H:%M").upper()),
    ):
        html = html.replace(token, value)
    return html


def build(path: Path | None = None) -> Path:
    conn = store.connect()
    df = build_frame(conn)
    if df.empty:
        raise analyze.NotEnoughData("No data cached yet — run `sync --full` first.")
    path = path or OUT_FILE
    path.write_text(render(df, conn), encoding="utf-8")
    return path
