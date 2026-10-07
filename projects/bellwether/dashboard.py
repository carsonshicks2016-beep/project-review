"""Bellwether dashboard — live view of the pipeline as it's built.

Design: each phase is a section in PHASES with a status() that reads the DB and
a render() that draws it. Sections auto-activate when their data exists, so
adding a later phase = fill in its render() and flip its status check. Nothing
else to wire.

Run:  make dashboard      (or: streamlit run dashboard.py)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from bellwether.config import DB_PATH, ROOT

PIDF = ROOT / "logs" / "watcher.pid"
LOGF = ROOT / "logs" / "watcher.log"
CAPTURES = ROOT / "data" / "audit_captures.jsonl"  # cloud (GitHub Actions) mode
WORKFLOW = ROOT / ".github" / "workflows" / "watcher-audit.yml"
AUDIT_HOURS = 48
AUTOSYNC_SECONDS = 300  # how often auto-sync runs `git pull` (when toggled on)


# -----------------------------------------------------------------------------
# shell plumbing — run gh/git/python ops from buttons
# -----------------------------------------------------------------------------
def _bin(name: str, fallbacks: list[str]) -> str:
    return shutil.which(name) or next(
        (f for f in fallbacks if Path(f).exists()), name
    )


GH = _bin("gh", ["/opt/homebrew/bin/gh", "/usr/local/bin/gh"])
GIT = _bin("git", ["/usr/bin/git", "/opt/homebrew/bin/git"])
PY = sys.executable


def run_cmd(args: list[str], timeout: int = 240) -> tuple[int, str]:
    try:
        r = subprocess.run(
            args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:  # timeout, missing binary, etc.
        return 1, str(e)


def cloud_mode() -> bool:
    return WORKFLOW.exists()


@st.cache_data(ttl=45, show_spinner=False)
def gh_runs(n: int = 6) -> list | None:
    code, out = run_cmd([GH, "run", "list", "--workflow=watcher-audit.yml",
                         "--limit", str(n), "--json",
                         "status,conclusion,createdAt,event,url"])
    if code != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


@st.cache_data(ttl=45, show_spinner=False)
def gh_state() -> str | None:
    code, out = run_cmd([GH, "workflow", "list", "--json", "name,state"])
    if code != 0:
        return None
    try:
        for w in json.loads(out):
            if "watcher-audit" in w.get("name", ""):
                return w.get("state")
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def repo_url() -> str | None:
    code, out = run_cmd([GIT, "remote", "get-url", "origin"])
    return out.replace(".git", "").strip() if code == 0 and out else None


def _clear_gh_cache() -> None:
    gh_runs.clear()
    gh_state.clear()

st.set_page_config(page_title="Bellwether", page_icon="🔔", layout="wide")


# -----------------------------------------------------------------------------
# data helpers
# -----------------------------------------------------------------------------
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(sql, c, params=params)


def count(table: str, where: str = "") -> int:
    try:
        return int(q(f"SELECT COUNT(*) n FROM {table} {where}")["n"].iloc[0])
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# phase sections
# -----------------------------------------------------------------------------
def status_pond() -> str:
    return "live" if count("companies", "WHERE in_universe=1") else "pending"


def render_pond() -> None:
    df = q(
        """SELECT ticker, name, sector,
                  round(mktcap/1e9,2) AS cap_B,
                  round(adv_usd/1e6,1) AS adv_M,
                  analyst_count AS analysts, notes
           FROM companies WHERE in_universe=1
           ORDER BY sector, analyst_count"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("In universe", len(df))
    c2.metric("Sectors", df["sector"].nunique() if len(df) else 0)
    c3.metric("Avg analysts", f"{df['analysts'].mean():.1f}" if len(df) else "—")
    c4.metric("Total screened", count("companies"))
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(df):
        st.caption("Sector mix")
        st.bar_chart(df.groupby("sector").size(), horizontal=True)


# -----------------------------------------------------------------------------
# 48-hour watcher audit (Step 2.11) — live panel
# -----------------------------------------------------------------------------
def audit_start() -> dt.datetime | None:
    """When the watch run began = mtime of the pid file (UTC)."""
    if PIDF.exists():
        return dt.datetime.fromtimestamp(PIDF.stat().st_mtime, tz=dt.timezone.utc)
    return None


def watcher_running() -> bool:
    try:
        pid = int(PIDF.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def poll_stats() -> tuple[int, str | None]:
    """(#polls completed, last poll timestamp string) from the log."""
    polls, last = 0, None
    if LOGF.exists():
        for ln in LOGF.read_text(errors="ignore").splitlines():
            if "poll complete" in ln:
                polls += 1
                m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
                if m:
                    last = m.group(1)
    return polls, last


def _captures_df() -> pd.DataFrame:
    """Filings captured during the audit, from whichever runner is active.

    Cloud mode (GitHub Actions) → data/audit_captures.jsonl.
    Local watch mode → DB rows fetched after the audit started.
    Returns columns incl. captured (UTC) + accepted (UTC).
    """
    if CAPTURES.exists():
        recs = [json.loads(l) for l in CAPTURES.read_text().splitlines() if l.strip()]
        caps = pd.DataFrame(recs)
        if len(caps):
            caps["captured"] = pd.to_datetime(caps["captured_at"], utc=True, errors="coerce")
            caps["accepted"] = pd.to_datetime(caps["acceptance_datetime"], utc=True, errors="coerce")
        return caps
    start = audit_start()
    if start is None:
        return pd.DataFrame()
    df = q(
        """SELECT c.ticker, f.form_type, f.acceptance_datetime, f.fetched_at
           FROM filings f JOIN companies c ON f.cik=c.cik
           ORDER BY f.fetched_at DESC LIMIT 100"""
    )
    if not len(df):
        return df
    df["captured"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    df["accepted"] = pd.to_datetime(df["acceptance_datetime"], utc=True, errors="coerce")
    return df[df["captured"] >= start].copy()


def _fmt_iso(s: str) -> str:
    try:
        return f"{pd.to_datetime(s, utc=True):%m-%d %H:%M}"
    except Exception:
        return s or "—"


def _maybe_autosync() -> None:
    """If auto-sync is on, `git pull` at most once per AUTOSYNC_SECONDS."""
    if not st.session_state.get("autosync"):
        return
    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    if now_ts - st.session_state.get("autosync_last", 0.0) >= AUTOSYNC_SECONDS:
        code, out = run_cmd([GIT, "pull", "--no-edit"])
        st.session_state["autosync_last"] = now_ts
        st.session_state["autosync_msg"] = (
            f"{dt.datetime.now():%H:%M:%S} "
            + ("✓ synced" if code == 0 else f"✗ {out[:60]}")
        )
        _clear_gh_cache()


@st.fragment(run_every=60)
def render_audit() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    _maybe_autosync()
    caps = _captures_df()

    if cloud_mode():
        state = gh_state()
        runs = gh_runs()
        last = runs[0] if runs else None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Runner", "☁️ GitHub Actions")
        c2.metric("Workflow", state or "—")
        c3.metric("Last run", (last["conclusion"] or last["status"]) if last else "—")
        c4.metric("Captured", len(caps))
        st.caption(
            "Polling in the cloud every ~15 min — laptop can be off. "
            "Click **Pull latest captures** below to sync this view."
        )
        if runs:
            rdf = pd.DataFrame([
                {"when": _fmt_iso(r["createdAt"]),
                 "result": r["conclusion"] or r["status"],
                 "trigger": r["event"]}
                for r in runs
            ])
            with st.expander(f"recent cloud runs ({len(runs)})", expanded=False):
                st.dataframe(rdf, use_container_width=True, hide_index=True)
        elif runs is None:
            st.caption("⚠️ couldn't reach GitHub CLI from the dashboard.")
    else:
        start, running = audit_start(), watcher_running()
        polls, last_poll = poll_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Watcher", "🟢 running" if running else "🔴 stopped")
        if start:
            elapsed_h = (now - start).total_seconds() / 3600
            c2.metric("Elapsed", f"{elapsed_h:.1f}h")
            c3.metric("Remaining", f"{max(AUDIT_HOURS - elapsed_h, 0):.1f}h")
            st.progress(min((now - start).total_seconds() / (AUDIT_HOURS * 3600), 1.0))
        c4.metric("Polls done", polls)

    if len(caps):
        caps["latency_min"] = (
            (caps["captured"] - caps["accepted"]).dt.total_seconds() / 60
        ).round(0)
        st.dataframe(
            caps[["ticker", "form_type", "acceptance_datetime", "latency_min"]]
            .sort_values("acceptance_datetime", ascending=False),
            use_container_width=True, hide_index=True,
        )
        med = caps["latency_min"].median()
        ok = "✅" if med < 15 else "⚠️"
        st.caption(f"{ok} median capture latency: {med:.0f} min  (gate: < 15 min)")
    else:
        st.info(
            "No new filings captured yet — these are quiet, neglected names, so "
            "captures are sparse. The cloud runner will catch any that file. "
            "(Materiality labeling can use the 440 already-backfilled filings "
            "anytime — see the Watcher tab.)"
        )
    sync = (
        f"auto-sync on · {st.session_state.get('autosync_msg', 'pending')}"
        if st.session_state.get("autosync") else "auto-sync off"
    )
    st.caption(f"auto-refreshes every 60s · {sync} · {now:%H:%M:%S} UTC")


def _do(label: str, args: list[str], clear_gh: bool = False) -> None:
    """Run a command, stash its result for display, refresh caches."""
    with st.spinner(f"{label}…"):
        code, out = run_cmd(args)
    st.session_state["cmd_result"] = (label, code, out)
    if clear_gh:
        _clear_gh_cache()


def render_controls() -> None:
    st.subheader("⚙️ Controls")

    st.toggle(
        "🔄 Auto-sync from cloud — `git pull` every 5 min while this tab is open",
        key="autosync",
        help="Pulls cloud-captured filings automatically so you don't have to "
             "click. Only runs while the dashboard tab is open.",
    )

    st.caption("Cloud audit (GitHub Actions)")
    a = st.columns(4)
    if a[0].button("⤵️ Pull latest captures", use_container_width=True,
                   help="git pull — sync cloud-captured filings into this dashboard"):
        _do("git pull", [GIT, "pull", "--no-edit"], clear_gh=True)
    if a[1].button("▶️ Trigger cloud poll", use_container_width=True,
                   help="Run the EDGAR poll in the cloud right now"):
        _do("gh workflow run", [GH, "workflow", "run", "watcher-audit.yml"], clear_gh=True)
    _state = gh_state()
    toggling = "disable" if _state == "active" else "enable"
    if a[2].button(f"{'⏸️ Disable' if _state=='active' else '▶️ Enable'} audit",
                   use_container_width=True,
                   help="Turn the scheduled cloud poll on/off"):
        _do(f"gh workflow {toggling}", [GH, "workflow", toggling, "watcher-audit.yml"],
            clear_gh=True)
    url = repo_url()
    if url:
        a[3].link_button("↗ Actions tab", f"{url}/actions", use_container_width=True)

    st.caption("Local / data")
    b = st.columns(3)
    if b[0].button("🔄 Refresh universe", use_container_width=True,
                   help="Re-pull metrics + re-screen the pond (scripts/load_universe.py)"):
        _do("refresh universe", [PY, "scripts/load_universe.py"])
    if b[1].button("📡 Local poll → DB", use_container_width=True,
                   help="One EDGAR sweep into the local DB (scripts/run_watcher.py poll)"):
        _do("local poll", [PY, "scripts/run_watcher.py", "poll"])
    if b[2].button("⬇️ Backfill recent", use_container_width=True,
                   help="Re-seed recent filing history (scripts/run_watcher.py backfill)"):
        _do("backfill", [PY, "scripts/run_watcher.py", "backfill"])

    if "cmd_result" in st.session_state:
        label, code, out = st.session_state["cmd_result"]
        (st.success if code == 0 else st.error)(f"`{label}` → exit {code}")
        if out:
            st.code(out[-3000:])


def status_watcher() -> str:
    return "live" if count("filings") else "pending"


def render_watcher() -> None:
    total = count("filings")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filings captured", total)
    c2.metric("With document", count("filings", "WHERE raw_path IS NOT NULL"))
    c3.metric("8-Ks", count("filings", "WHERE form_type='8-K'"))
    c4.metric("Form 4 (insider)", count("filings", "WHERE form_type='4'"))

    left, right = st.columns(2)
    with left:
        st.caption("By form type (top 10)")
        forms = q(
            """SELECT form_type, COUNT(*) n FROM filings
               GROUP BY form_type ORDER BY n DESC LIMIT 10"""
        ).set_index("form_type")
        st.bar_chart(forms, horizontal=True)
    with right:
        st.caption("By company")
        byco = q(
            """SELECT ticker, COUNT(*) n FROM filings f
               JOIN companies c ON f.cik=c.cik
               GROUP BY ticker ORDER BY n DESC"""
        ).set_index("ticker")
        st.bar_chart(byco)

    st.caption("Most recent filings")
    recent = q(
        """SELECT c.ticker, f.form_type, f.filed_date,
                  f.acceptance_datetime, f.accession_no
           FROM filings f JOIN companies c ON f.cik=c.cik
           ORDER BY f.acceptance_datetime DESC LIMIT 25"""
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)


def status_signals() -> str:
    return "live" if count("signals") else "pending"


def render_signals() -> None:
    if count("signals") == 0:
        _coming(
            "Phase 3 — The Extractor",
            "An LLM reads each new filing and scores it: catalyst type, "
            "materiality (0–10), direction, and the exact sentence behind the "
            "call. This panel lights up automatically once the `signals` table "
            "has rows.",
        )
        return
    df = q(
        """SELECT c.ticker, s.catalyst_type, s.direction, s.materiality,
                  s.confidence, s.summary, s.evidence_quote, s.created_at
           FROM signals s JOIN filings f ON s.filing_id=f.id
           JOIN companies c ON f.cik=c.cik
           ORDER BY s.materiality DESC, s.created_at DESC"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signals", len(df))
    c2.metric("High materiality (≥7)", int((df["materiality"] >= 7).sum()))
    c3.metric("Bullish", int((df["direction"] == "bullish").sum()))
    c4.metric("Bearish", int((df["direction"] == "bearish").sum()))

    # Filters
    with st.expander("🔍 Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        direction_filter = f1.multiselect(
            "Direction", ["bullish", "bearish", "neutral"],
            default=["bullish", "bearish", "neutral"]
        )
        catalyst_options = sorted(df["catalyst_type"].unique().tolist())
        catalyst_filter = f2.multiselect("Catalyst Type", catalyst_options, default=catalyst_options)
        mat_min = f3.slider("Min Materiality", 0, 10, 0)

    filtered = df[
        df["direction"].isin(direction_filter) &
        df["catalyst_type"].isin(catalyst_filter) &
        (df["materiality"] >= mat_min)
    ]
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def status_backtest() -> str:
    return "live" if count("outcomes") else "pending"


def render_backtest() -> None:
    if count("outcomes") == 0:
        _coming(
            "Phase 4 — The Backtest",
            "The make-or-break gate: replay historical filings at their public "
            "timestamps and measure whether high-materiality flags preceded real "
            "abnormal returns — defeating look-ahead, survivorship, and slippage. "
            "Activates when the `outcomes` table is populated.",
        )
        return

    df = q(
        """SELECT s.catalyst_type, s.direction,
                  o.fwd_ret_1d, o.fwd_ret_5d, o.fwd_ret_20d, o.abnormal_ret_5d,
                  o.benchmark
           FROM outcomes o JOIN signals s ON o.signal_id = s.id"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Backtested Signals", len(df))
    if df["abnormal_ret_5d"].notna().any():
        mean_ab = df["abnormal_ret_5d"].mean()
        c2.metric("Mean Abnormal Ret (+5d)", f"{mean_ab*100:+.2f}%")
        hits = int((df["abnormal_ret_5d"] > 0).sum())
        c3.metric("Hit Rate", f"{hits}/{len(df)} ({hits/len(df)*100:.0f}%)")

    st.divider()
    st.subheader("Forward Returns by Catalyst")
    pivot = df.groupby("catalyst_type")[["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"]].mean() * 100
    if not pivot.empty:
        st.bar_chart(pivot, use_container_width=True)

    st.subheader("Raw Outcomes")
    display_df = df.copy()
    for col in ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d", "abnormal_ret_5d"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def status_alerts() -> str:
    try:
        return "live" if count("alerts_sent") else "pending"
    except Exception:
        return "pending"


def render_alerts() -> None:
    try:
        n_sent = count("alerts_sent")
    except Exception:
        n_sent = 0

    if n_sent == 0:
        _coming(
            "Phase 5 — Alerts",
            "High-conviction signals (materiality ≥ 7) push to email with the "
            "summary, score, evidence quote, and a link — within minutes of the "
            "filing. This tab lights up once the first alert is sent.",
        )
        return

    df = q(
        """SELECT s.id as signal_id, c.ticker, s.catalyst_type, s.direction,
                  s.materiality, a.sent_at
           FROM alerts_sent a
           JOIN signals s ON a.signal_id = s.id
           JOIN filings f ON s.filing_id = f.id
           JOIN companies c ON f.cik = c.cik
           ORDER BY a.sent_at DESC"""
    )
    st.metric("Alerts Sent", len(df))
    st.dataframe(df, use_container_width=True, hide_index=True)


def _coming(title: str, body: str) -> None:
    st.info(f"**{title}** — not built yet.\n\n{body}")


# -----------------------------------------------------------------------------
# phase registry  (add a later phase = append a row here)
# -----------------------------------------------------------------------------
PHASES = [
    ("1", "Pond", status_pond, render_pond),
    ("2", "Watcher", status_watcher, render_watcher),
    ("3", "Signals", status_signals, render_signals),
    ("4", "Backtest", status_backtest, render_backtest),
    ("5", "Alerts", status_alerts, render_alerts),
]

BADGE = {"live": "🟢 live", "pending": "⚪ pending"}


# -----------------------------------------------------------------------------
# layout
# -----------------------------------------------------------------------------
st.title("🔔 Bellwether")
st.caption(
    "Coverage-gap intelligence for neglected equities — reading the filings "
    "Wall Street ignores. Personal tool (v1)."
)

if not DB_PATH.exists():
    st.error(f"No database at {DB_PATH}. Run `make init-db` and load the pond first.")
    st.stop()

# phase progress strip
cols = st.columns(len(PHASES))
for col, (num, name, status, _) in zip(cols, PHASES):
    s = status()
    col.metric(f"Phase {num}", name, BADGE[s])

st.divider()

# live 48-hour audit banner (cloud workflow OR local watch run OR captures present)
if cloud_mode() or PIDF.exists() or CAPTURES.exists():
    with st.container(border=True):
        st.subheader("🔴 Live watcher audit · Step 2.11 (M1 gate)")
        render_audit()
    with st.container(border=True):
        render_controls()
    st.divider()

# one tab per phase
tabs = st.tabs([f"Phase {n} · {nm}  {BADGE[stat()].split()[0]}"
                for n, nm, stat, _ in PHASES])
for tab, (num, name, status, render) in zip(tabs, PHASES):
    with tab:
        render()

st.divider()
st.caption(
    f"Reading {DB_PATH.name} · refreshed {dt.datetime.now():%Y-%m-%d %H:%M:%S} · "
    "PLAN.md & BUILD_ROADMAP.md are authoritative"
)
