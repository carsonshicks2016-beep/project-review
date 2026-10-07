"""Standalone EDGAR poll for the cloud (GitHub Actions) — no local DB.

Reads data/universe.csv (ticker,cik), polls each company's EDGAR submissions
feed, and appends any filing NOT already known to data/audit_captures.jsonl,
stamped with captured_at (UTC now) so latency = captured_at − acceptance_datetime.

State persists in the committed files, so each scheduled run sees what prior runs
captured:
  - data/audit_baseline.txt   accessions known before the audit (don't log these)
  - data/audit_captures.jsonl filings captured DURING the audit (append-only)

No API keys, no database. Requires only SEC_USER_AGENT in the environment.
"""

from __future__ import annotations

import csv
import datetime as dt
import json

from bellwether import sec
from bellwether.config import ROOT

UNIVERSE = ROOT / "data" / "universe.csv"
BASELINE = ROOT / "data" / "audit_baseline.txt"
CAPTURES = ROOT / "data" / "audit_captures.jsonl"
POLL_DEPTH = 15  # newest N per company; new filings appear at the top


def known_accessions() -> set[str]:
    seen: set[str] = set()
    if BASELINE.exists():
        seen |= {a.strip() for a in BASELINE.read_text().splitlines() if a.strip()}
    if CAPTURES.exists():
        for line in CAPTURES.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["accession_no"])
    return seen


def main() -> None:
    rows = list(csv.DictReader(UNIVERSE.open()))
    seen = known_accessions()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    new = []
    for r in rows:
        cik, ticker = r["cik"].strip(), r["ticker"].strip()
        try:
            subs = sec.fetch_submissions(cik)
        except Exception as e:  # network / rate / symbol
            print(f"WARN {ticker}: {e}")
            continue
        for f in sec.parse_recent(subs, cik)[:POLL_DEPTH]:
            if f["accession_no"] not in seen:
                seen.add(f["accession_no"])
                new.append({
                    "ticker": ticker,
                    "cik": f["cik"],
                    "form_type": f["form_type"],
                    "accession_no": f["accession_no"],
                    "filed_date": f["filed_date"],
                    "acceptance_datetime": f["acceptance_datetime"],
                    "url": f["url"],
                    "captured_at": now,
                })
    if new:
        with CAPTURES.open("a") as fh:
            for rec in new:
                fh.write(json.dumps(rec) + "\n")
        for rec in new:
            print(f"NEW {rec['ticker']:<5} {rec['form_type']:<8} {rec['accession_no']}")
    print(f"poll complete: {len(new)} new filings ({len(seen)} known)")


if __name__ == "__main__":
    main()
