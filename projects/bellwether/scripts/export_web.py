"""Export the live DB to JSON snapshots the web frontend reads.

Writes bellwether-web/public/data/*.json. Run locally, or from the cloud Action
so the static site always has fresh data with no backend.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
import subprocess

from bellwether.config import DB_PATH, ROOT

OUT = ROOT / "bellwether-web" / "public" / "data"
CAPTURES = ROOT / "data" / "audit_captures.jsonl"
GH = shutil.which("gh") or "/opt/homebrew/bin/gh"
GIT = shutil.which("git") or "/usr/bin/git"


def _run(args: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=40)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return 1, str(e)


def _audit_snapshot() -> dict:
    """Captures (from the cloud's committed JSONL) + a GitHub status snapshot."""
    captures = []
    if CAPTURES.exists():
        for line in CAPTURES.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            lat = None
            try:
                cap = dt.datetime.fromisoformat(r["captured_at"].replace("Z", "+00:00"))
                acc = dt.datetime.fromisoformat(r["acceptance_datetime"].replace("Z", "+00:00"))
                lat = round((cap - acc).total_seconds() / 60)
            except Exception:
                pass
            captures.append({"ticker": r["ticker"], "form_type": r["form_type"],
                             "acceptance_datetime": r["acceptance_datetime"],
                             "captured_at": r["captured_at"], "latency_min": lat})

    state, runs, url = None, [], None
    code, out = _run([GH, "workflow", "list", "--json", "name,state"])
    if code == 0:
        try:
            state = next((w["state"] for w in json.loads(out)
                          if "watcher-audit" in w.get("name", "")), None)
        except Exception:
            pass
    code, out = _run([GH, "run", "list", "--workflow=watcher-audit.yml", "--limit", "8",
                      "--json", "status,conclusion,createdAt,event,displayTitle"])
    if code == 0:
        try:
            runs = json.loads(out)
        except Exception:
            pass
    code, out = _run([GIT, "remote", "get-url", "origin"])
    if code == 0 and out:
        url = out.replace(".git", "").strip()

    return {
        "state": state,
        "repo_url": url,
        "captures": sorted(captures, key=lambda c: c["acceptance_datetime"], reverse=True),
        "runs": runs,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    q = lambda s: [dict(r) for r in c.execute(s).fetchall()]
    one = lambda s: c.execute(s).fetchone()[0]

    pond = q(
        """SELECT ticker, name, sector, mktcap, adv_usd, analyst_count, notes
           FROM companies WHERE in_universe=1 ORDER BY sector, analyst_count"""
    )
    by_form = q(
        "SELECT form_type, COUNT(*) n FROM filings GROUP BY form_type ORDER BY n DESC LIMIT 10"
    )
    by_co = q(
        """SELECT ticker, COUNT(*) n FROM filings f JOIN companies c ON f.cik=c.cik
           GROUP BY ticker ORDER BY n DESC"""
    )
    recent = q(
        """SELECT c.ticker, f.form_type, f.filed_date, f.acceptance_datetime
           FROM filings f JOIN companies c ON f.cik=c.cik
           ORDER BY f.acceptance_datetime DESC LIMIT 30"""
    )
    analysts = [p["analyst_count"] for p in pond if p["analyst_count"] is not None]

    overview = {
        "pond": len(pond),
        "screened": one("SELECT COUNT(*) FROM companies"),
        "filings": one("SELECT COUNT(*) FROM filings"),
        "sectors": len({p["sector"] for p in pond}),
        "avg_analysts": round(sum(analysts) / len(analysts), 1) if analysts else None,
        "signals": one("SELECT COUNT(*) FROM signals"),
    }

    # Signals: real rows if any, else clearly-labelled previews so the signal-feed
    # motion is visible before Phase 3 ships.
    signals = q(
        """SELECT c.ticker, s.catalyst_type, s.direction, s.materiality,
                  s.confidence, s.summary, s.evidence_quote, s.created_at
           FROM signals s JOIN filings f ON s.filing_id=f.id
           JOIN companies c ON f.cik=c.cik
           ORDER BY s.materiality DESC"""
    )
    preview = not signals
    if preview:
        signals = [
            {"ticker": "HPK", "catalyst_type": "new contract", "direction": "bullish",
             "materiality": 8, "summary": "Signed a multi-year supply agreement materially above prior guidance.",
             "evidence_quote": "…a five-year take-or-pay contract representing approximately 40% of annual production…",
             "form_type": "8-K"},
            {"ticker": "DBD", "catalyst_type": "impairment", "direction": "bearish",
             "materiality": 6, "summary": "Disclosed a non-cash goodwill impairment in its retail segment.",
             "evidence_quote": "…recorded a goodwill impairment charge of $48.2 million during the quarter…",
             "form_type": "8-K"},
            {"ticker": "SD", "catalyst_type": "insider buy cluster", "direction": "bullish",
             "materiality": 7, "summary": "Three insiders bought on the open market within a week.",
             "evidence_quote": "…acquired 25,000 shares at a weighted average price of $13.90…",
             "form_type": "Form 4"},
        ]

    (OUT / "overview.json").write_text(json.dumps(overview, indent=2))
    (OUT / "universe.json").write_text(json.dumps(pond, indent=2))
    (OUT / "filings.json").write_text(json.dumps(
        {"by_form": by_form, "by_company": by_co, "recent": recent}, indent=2))
    (OUT / "signals.json").write_text(json.dumps(
        {"preview": preview, "items": signals}, indent=2))

    audit = _audit_snapshot()
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))

    print(f"audit: {len(audit['captures'])} captures, "
          f"workflow {audit['state']}, {len(audit['runs'])} recent runs")
    print(f"exported → {OUT.relative_to(ROOT)}: "
          f"{len(pond)} pond, {overview['filings']} filings, "
          f"{len(signals)} signals{' (preview)' if preview else ''}")


if __name__ == "__main__":
    main()
